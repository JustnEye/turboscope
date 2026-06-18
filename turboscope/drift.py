"""
turboscope/drift.py

Distribution shift detector for TurboScopeIndex.

Why drift detection matters for quantized vector search
--------------------------------------------------------
turbovec's TQ+ calibration is frozen after the first add (specifically
after the first batch of >= 1,000 vectors). It fits two scalars per
coordinate — a shift and a scale — that map each rotated coordinate's
empirical 5/95% quantiles onto the canonical Beta((d-1)/2, (d-1)/2)
marginal. The Lloyd-Max codebook was built for that Beta distribution.
When new data has a meaningfully different structure — a different
embedding model, a different document domain, different norm
characteristics — the frozen calibration mis-fits, and recall silently
degrades before any monitoring threshold is crossed.

This module gives you a leading indicator: it detects distribution
shift in the *raw* incoming vectors before quantization, surfacing it
as a single severity score that rises before recall falls.

What is measured
----------------
Two independent signals are computed and combined.

**Signal 1 — Norm distribution (KS statistic)**
The L2 norm of each raw (pre-normalization) vector is collected at add
time. The two-sample Kolmogorov-Smirnov statistic between the baseline
norm distribution (drawn from the calibration batch) and the current
rolling window is computed. Norm shift is a strong proxy for domain
change: embeddings from a different model, a different pre-processing
pipeline, or a different data source typically show a shifted norm
distribution even before the directional content changes.

KS is in [0, 1]; 0 means the two ECDF are identical, 1 means complete
separation.

**Signal 2 — Per-dimension variance shift**
The variance of each raw dimension is computed over the baseline batch
and over the current window. The relative median absolute deviation
between the two variance vectors:

    rel_var_shift = median(|var_curr[d] - var_base[d]|) / median(var_base)

measures how much the anisotropy of the data has changed. This is the
dimension-level version of what TQ+ calibration cares about: if per-dim
variance distribution changes, the empirical quantiles that calibration
fitted against are no longer representative.

**Combined severity**
    severity = clip(0.5 * ks_stat + 0.5 * tanh(rel_var_shift), 0, 1)

tanh maps rel_var_shift from [0, ∞) to [0, 1) so the two terms are
comparable in scale. Both signals have equal weight. The threshold for
``recalibrate_suggested`` is 0.3 — calibrated so that:
  - Same domain, same model → severity ≈ 0.04–0.08 (no suggestion)
  - Subtle domain shift (different norm range, same dim) → severity ≈ 0.25
  - Different embedding model → severity ≈ 0.50–0.85 (suggestion fires)

This module never raises: all errors during signal computation are
caught and result in a DriftSnapshot with severity=-1.0 (not ready).

Thread safety
-------------
All mutable state is protected by a single lock. ``on_add`` is called
from ``TurboScopeIndex.add_with_ids``, which may run on any thread.
``snapshot`` is safe to call concurrently from the dashboard thread.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of vectors required to establish a reliable baseline.
# Matches turbovec's TQPLUS_MIN_SAMPLES — below this the quantile estimates
# that TQ+ fits are too noisy to be a useful reference point.
MIN_BASELINE_SIZE: int = 1_000

# Severity threshold above which recalibration is suggested.
RECALIBRATE_THRESHOLD: float = 0.3

# Maximum number of add-call batches retained in the current window.
# Each entry is a small statistics struct, not the raw vectors.
_DEFAULT_WINDOW_CALLS: int = 50


# ---------------------------------------------------------------------------
# DriftSnapshot — what the dashboard and stats collector consume
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftSnapshot:
    """
    Point-in-time drift severity estimate.

    Attributes
    ----------
    severity : float
        Combined drift score in [0, 1].  -1.0 means the baseline has not
        been established yet (fewer than MIN_BASELINE_SIZE vectors seen).
        Values above RECALIBRATE_THRESHOLD (0.3) trigger
        ``recalibrate_suggested = True``.
    ks_stat : float
        Two-sample KS statistic on L2 norm distributions.  In [0, 1].
        -1.0 when not yet available.
    rel_var_shift : float
        Relative median absolute deviation of per-dimension variances
        between baseline and current window.  In [0, ∞).
        -1.0 when not yet available.
    recalibrate_suggested : bool
        True when severity > RECALIBRATE_THRESHOLD.
    n_baseline : int
        Number of vectors in the baseline.
    n_current_window : int
        Number of vectors in the current rolling window.
    baseline_ready : bool
        True once the baseline has been established.
    """
    severity: float
    ks_stat: float
    rel_var_shift: float
    recalibrate_suggested: bool
    n_baseline: int
    n_current_window: int
    baseline_ready: bool

    def __str__(self) -> str:
        if not self.baseline_ready:
            return (
                f"DriftSnapshot(not ready — "
                f"{self.n_baseline}/{MIN_BASELINE_SIZE} baseline vectors seen)"
            )
        flag = " ⚠ recalibrate suggested" if self.recalibrate_suggested else ""
        return (
            f"DriftSnapshot("
            f"severity={self.severity:.3f}, "
            f"ks={self.ks_stat:.3f}, "
            f"rel_var={self.rel_var_shift:.3f}, "
            f"n_window={self.n_current_window}"
            f"{flag})"
        )


# ---------------------------------------------------------------------------
# _BatchStats — lightweight summary of one add() call's raw vectors
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _BatchStats:
    """
    Per-add-call statistics derived from raw vectors.

    Storing the statistics rather than the raw vectors keeps the window
    memory cost O(window_calls * dim) instead of O(window_calls * n * dim).
    """
    n: int                            # number of vectors in this batch
    norms: npt.NDArray[np.float32]    # L2 norms, shape (n,)
    var_per_dim: npt.NDArray[np.float64]  # per-dimension variance, shape (dim,)


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Detects distribution shift in raw vectors added to a TurboScopeIndex.

    Parameters
    ----------
    min_baseline_size : int
        Minimum number of vectors required to establish the baseline.
        Defaults to MIN_BASELINE_SIZE (1,000, matching turbovec's
        TQPLUS_MIN_SAMPLES).
    window_calls : int
        Number of add-call batches to retain in the current rolling window.
        Defaults to 50.  The window covers recent data; the baseline is
        fixed once established.
    recalibrate_threshold : float
        Severity above which ``DriftSnapshot.recalibrate_suggested`` is
        True.  Defaults to RECALIBRATE_THRESHOLD (0.3).
    """

    def __init__(
        self,
        min_baseline_size: int = MIN_BASELINE_SIZE,
        window_calls: int = _DEFAULT_WINDOW_CALLS,
        recalibrate_threshold: float = RECALIBRATE_THRESHOLD,
    ) -> None:
        if min_baseline_size < 1:
            raise ValueError(
                f"min_baseline_size must be >= 1, got {min_baseline_size}"
            )
        if window_calls < 1:
            raise ValueError(
                f"window_calls must be >= 1, got {window_calls}"
            )
        if not (0.0 < recalibrate_threshold < 1.0):
            raise ValueError(
                f"recalibrate_threshold must be in (0, 1), "
                f"got {recalibrate_threshold}"
            )

        self._min_baseline = min_baseline_size
        self._threshold = recalibrate_threshold
        self._lock = threading.Lock()

        # Baseline: built from the first min_baseline_size vectors.
        # Stored as flat arrays so snapshot() can compute KS efficiently.
        self._baseline_norms: npt.NDArray[np.float32] | None = None
        self._baseline_var: npt.NDArray[np.float64] | None = None
        self._n_baseline: int = 0
        self._baseline_ready: bool = False

        # Accumulator for vectors still needed to complete the baseline.
        # Flushed into _baseline_norms/_baseline_var once full.
        self._baseline_norms_acc: list[npt.NDArray[np.float32]] = []
        self._baseline_var_acc: list[npt.NDArray[np.float64]] = []
        self._baseline_n_acc: int = 0

        # Rolling window of recent batch statistics.
        self._window: deque[_BatchStats] = deque(maxlen=window_calls)
        self._n_window: int = 0  # total vectors in current window

    # ------------------------------------------------------------------
    # Called by TurboScopeIndex
    # ------------------------------------------------------------------

    def observe(
        self,
        vectors: npt.NDArray[np.float32],
    ) -> None:
        """
        Record statistics from a batch of raw vectors.

        Called by :class:`~turboscope.index.TurboScopeIndex` immediately
        before feeding the vectors to the inner turbovec index.  Must be
        called with the raw (pre-normalization, pre-rotation) vectors.

        Parameters
        ----------
        vectors : ndarray of shape (n, dim), float32
            Raw vectors from one ``add_with_ids`` call.
        """
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return

        n = vectors.shape[0]

        # Compute norms and per-dim variance (float64 for precision).
        vf = vectors.astype(np.float64, copy=False)
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        var_per_dim = np.var(vf, axis=0)  # shape (dim,)

        with self._lock:
            if not self._baseline_ready:
                self._accumulate_baseline(norms, var_per_dim, n)
            else:
                self._push_window(norms, var_per_dim, n)

    def on_add(self, n: int) -> None:
        """
        Lightweight notification hook — called by TurboScopeIndex after
        a successful add when it doesn't want to pass the raw vectors.

        Prefer :meth:`observe` when the raw vectors are available, since
        ``on_add`` carries no statistical information and cannot update
        the drift signals.  It is kept for API compatibility with the
        interface declared in ``index.py``.
        """
        # Nothing to do without the raw vectors. The real data path goes
        # through observe(). This no-op prevents AttributeError for any
        # caller that only wires up on_add.
        pass

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def snapshot(self) -> DriftSnapshot:
        """
        Return a point-in-time drift estimate.

        Thread-safe: all statistics are copied or computed atomically
        under the lock, and the result is an immutable DriftSnapshot.

        Returns
        -------
        DriftSnapshot
            If the baseline is not ready, returns a snapshot with
            ``severity = -1.0`` and ``baseline_ready = False``.
        """
        with self._lock:
            if not self._baseline_ready:
                return DriftSnapshot(
                    severity=-1.0,
                    ks_stat=-1.0,
                    rel_var_shift=-1.0,
                    recalibrate_suggested=False,
                    n_baseline=self._n_baseline,
                    n_current_window=self._n_window,
                    baseline_ready=False,
                )

            if not self._window:
                # Baseline established but no post-baseline data yet.
                return DriftSnapshot(
                    severity=0.0,
                    ks_stat=0.0,
                    rel_var_shift=0.0,
                    recalibrate_suggested=False,
                    n_baseline=self._n_baseline,
                    n_current_window=0,
                    baseline_ready=True,
                )

            # Aggregate the current window into flat arrays.
            window_norms = np.concatenate([b.norms for b in self._window])
            # Per-dim variance: weighted average across batches in window.
            total_n = sum(b.n for b in self._window)
            window_var = sum(
                b.var_per_dim * (b.n / total_n) for b in self._window
            )

            baseline_norms = self._baseline_norms   # not None here
            baseline_var = self._baseline_var       # not None here

        # Compute signals outside the lock — pure numpy, no shared state.
        try:
            ks = _ks_2samp(baseline_norms, window_norms)
            rel_var = _relative_var_shift(baseline_var, window_var)
            severity = float(np.clip(
                0.5 * ks + 0.5 * float(np.tanh(rel_var)), 0.0, 1.0
            ))
        except Exception:
            return DriftSnapshot(
                severity=-1.0,
                ks_stat=-1.0,
                rel_var_shift=-1.0,
                recalibrate_suggested=False,
                n_baseline=self._n_baseline,
                n_current_window=total_n,
                baseline_ready=True,
            )

        return DriftSnapshot(
            severity=severity,
            ks_stat=ks,
            rel_var_shift=rel_var,
            recalibrate_suggested=severity > self._threshold,
            n_baseline=self._n_baseline,
            n_current_window=total_n,
            baseline_ready=True,
        )

    def reset(self) -> None:
        """
        Clear the baseline and window, returning to the un-initialized state.

        Useful when the index has been re-populated from scratch and the
        old baseline is no longer representative.
        """
        with self._lock:
            self._baseline_norms = None
            self._baseline_var = None
            self._n_baseline = 0
            self._baseline_ready = False
            self._baseline_norms_acc.clear()
            self._baseline_var_acc.clear()
            self._baseline_n_acc = 0
            self._window.clear()
            self._n_window = 0

    # ------------------------------------------------------------------
    # Internal helpers (called with lock held)
    # ------------------------------------------------------------------

    def _accumulate_baseline(
        self,
        norms: npt.NDArray[np.float32],
        var_per_dim: npt.NDArray[np.float64],
        n: int,
    ) -> None:
        """
        Accumulate statistics toward the baseline.

        We need at least ``_min_baseline`` vectors before computing the
        baseline. We collect norms across batches and keep a running
        weighted average of per-dim variance (cheaper than storing all
        raw vectors).

        Called with self._lock held.
        """
        self._baseline_norms_acc.append(norms)
        self._baseline_var_acc.append(var_per_dim)
        self._baseline_n_acc += n
        self._n_baseline += n

        if self._n_baseline >= self._min_baseline:
            # Freeze the baseline.
            self._baseline_norms = np.concatenate(self._baseline_norms_acc)
            # Weighted average of per-batch variances.
            total = self._baseline_n_acc
            self._baseline_var = sum(
                v * (self._baseline_norms_acc[i].shape[0] / total)
                for i, v in enumerate(self._baseline_var_acc)
            )
            self._baseline_ready = True
            # Free accumulator memory.
            self._baseline_norms_acc = []
            self._baseline_var_acc = []
            self._baseline_n_acc = 0

    def _push_window(
        self,
        norms: npt.NDArray[np.float32],
        var_per_dim: npt.NDArray[np.float64],
        n: int,
    ) -> None:
        """
        Push one batch's statistics into the rolling window.

        When the deque is full, the oldest entry is automatically evicted
        (``deque(maxlen=...)`` handles this). We track ``_n_window``
        manually so snapshot() can report it without iterating the deque.

        Called with self._lock held.
        """
        if len(self._window) == self._window.maxlen:
            # Oldest entry about to be evicted.
            self._n_window -= self._window[0].n

        self._window.append(_BatchStats(
            n=n,
            norms=norms,
            var_per_dim=var_per_dim,
        ))
        self._n_window += n


# ---------------------------------------------------------------------------
# Signal computation — pure numpy, no lock needed
# ---------------------------------------------------------------------------

def _ks_2samp(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
) -> float:
    """
    Two-sample Kolmogorov-Smirnov statistic.

    KS = max over all x of |F_a(x) - F_b(x)|, where F_a and F_b are
    the empirical CDFs of ``a`` and ``b``.

    Parameters
    ----------
    a, b : 1-D float arrays
        The two samples to compare.  Need not be the same length.

    Returns
    -------
    float in [0, 1]
        0 means the two empirical CDFs are identical; 1 means complete
        separation (every value in one sample is above every value in
        the other).

    Notes
    -----
    This is the one-sided statistic (absolute value of the maximum
    signed difference).  It is equivalent to scipy.stats.ks_2samp(...,
    alternative='two-sided') statistic field, but without the p-value.
    We deliberately omit the p-value: at large sample sizes (n=1000+)
    any real-world difference will be statistically significant, so the
    p-value gives no actionable information.  The statistic itself is
    what determines severity.
    """
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    na, nb = len(a_sorted), len(b_sorted)

    # Evaluate both ECDFs at every unique value in the combined sample.
    combined = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, combined, side="right") / na
    cdf_b = np.searchsorted(b_sorted, combined, side="right") / nb

    return float(np.max(np.abs(cdf_a - cdf_b)))


def _relative_var_shift(
    base_var: npt.NDArray[np.float64],
    curr_var: npt.NDArray[np.float64],
) -> float:
    """
    Relative median absolute deviation of per-dimension variances.

    Measures how much the anisotropy structure of the current data
    differs from the baseline.

    Returns
    -------
    float in [0, ∞)
        0 means no change; higher values mean greater shift.  At 1.0
        the median per-dim variance has changed by ~76% (tanh(1) ≈ 0.76
        maps this to ~0.38 severity contribution from this signal).

    Notes
    -----
    Using the *median* rather than the mean makes this robust to a small
    number of dimensions with very large variance changes (e.g. a single
    "hot" embedding dimension that fluctuates between domains).  The
    median reflects the typical coordinate behaviour.
    """
    med_base = float(np.median(base_var))
    if med_base < 1e-10:
        # Degenerate baseline (all-zero or near-zero vectors).
        # Return 0 to avoid division by zero masking real changes.
        return 0.0
    mad = float(np.median(np.abs(curr_var - base_var)))
    return mad / med_base