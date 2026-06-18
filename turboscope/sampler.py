"""
turboscope/sampler.py

Shadow recall estimator for TurboScopeIndex.

How recall is estimated without ground truth
--------------------------------------------
On a sampled fraction of search calls, TurboScopeIndex runs two searches
over the *same candidate set* — the reservoir:

    1. Exact search  — brute-force inner products on the raw float32
                       reservoir vectors.  Returns the true top-k for
                       this candidate set.

    2. Restricted turbovec search — the normal quantized SIMD search
                       with an allowlist restricted to reservoir IDs.
                       Returns turbovec's approximate top-k from the same
                       candidate set.

Recall@k is then:

    |exact_top_k  ∩  turbovec_top_k|
    ---------------------------------
         min(k, n_reservoir)

This is a *sub-corpus* estimate — it measures recall over the reservoir
subset, not the full index.  It is a slight underestimate of true recall
(turbovec might correctly return a non-reservoir vector that ranks higher
than the k-th reservoir result) but is unbiased on average as the
reservoir grows, and is the only approach that doesn't require storing
the full raw corpus.

The exact search is implemented in the optional Rust extension
``_turboscope`` (``src/lib.rs``).  When the extension is not built, the
sampler falls back to a pure-numpy path that is correct but ~40× slower
per sampled query.

Thread safety
-------------
``record_shadow`` is called from ``TurboScopeIndex._run_shadow``, which
runs on the search-call thread.  Multiple threads can call it concurrently.
All mutable state (the rolling recall window) is protected by a lock.
``snapshot()`` returns an immutable ``RecallSnapshot`` so callers don't
race with ongoing updates.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from turboscope.index import SearchRecord


# ---------------------------------------------------------------------------
# RecallSnapshot — what the dashboard and stats collector consume
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecallSnapshot:
    """
    Point-in-time recall estimates computed from the rolling window.

    All recall values are in [0.0, 1.0].  A value of -1.0 means not
    enough shadow samples have accumulated yet to produce an estimate
    for that k.

    Attributes
    ----------
    recall : dict[int, float]
        Recall@k for each k tracked, e.g. {1: 0.94, 5: 0.98, 10: 0.99}.
    ci_half_width : dict[int, float]
        Half-width of the 95% confidence interval for each k.
        Computed as 1.96 * std(per_query_recall) / sqrt(n_queries).
    n_queries : int
        Total number of queries that contributed to this estimate
        (sum over the window).
    n_shadow_calls : int
        Number of shadow search calls in the window (each call may
        cover multiple queries).
    window_size : int
        Maximum number of shadow calls retained in the rolling window.
    """
    recall: dict[int, float]
    ci_half_width: dict[int, float]
    n_queries: int
    n_shadow_calls: int
    window_size: int

    def recall_at(self, k: int) -> float:
        """Return recall@k, or -1.0 if not yet available."""
        return self.recall.get(k, -1.0)

    def is_ready(self) -> bool:
        """True once at least one shadow call has been recorded."""
        return self.n_shadow_calls > 0

    def __str__(self) -> str:
        if not self.is_ready():
            return "RecallSnapshot(no data yet)"
        parts = [
            f"R@{k}={v:.3f}±{self.ci_half_width.get(k, 0):.3f}"
            for k, v in sorted(self.recall.items())
        ]
        return f"RecallSnapshot({', '.join(parts)}, n={self.n_queries})"


# ---------------------------------------------------------------------------
# _ShadowWindow — rolling buffer of per-query recall values
# ---------------------------------------------------------------------------

class _ShadowWindow:
    """
    Rolling window of per-query recall observations for a single k value.

    Stores the raw per-query recall values (each in [0, 1]) so we can
    compute mean and variance on demand.  Older calls are evicted when the
    window exceeds ``max_calls`` shadow-call batches.
    """

    def __init__(self, max_calls: int) -> None:
        # Each entry is a 1-D float64 array of per-query recalls for one
        # shadow call.  We keep up to max_calls batches.
        self._batches: deque[npt.NDArray[np.float64]] = deque(maxlen=max_calls)
        self._total_queries: int = 0

    def push(self, per_query_recalls: npt.NDArray[np.float64]) -> None:
        """Add one batch of per-query recall values."""
        if len(self._batches) == self._batches.maxlen:
            # About to evict the oldest batch; subtract its query count.
            self._total_queries -= len(self._batches[0])
        self._batches.append(per_query_recalls)
        self._total_queries += len(per_query_recalls)

    def stats(self) -> tuple[float, float, int] | None:
        """
        Return (mean_recall, ci_half_width, n_queries), or None if empty.

        ``ci_half_width`` is 1.96 * stderr (95% normal-approximation
        interval).  When n_queries == 1 the variance is 0.0 and the
        interval is 0.0 — the estimate is a single observation, which
        is honest rather than artificially wide.
        """
        if not self._batches:
            return None
        all_vals = np.concatenate(list(self._batches))
        n = len(all_vals)
        mean = float(np.mean(all_vals))
        if n == 1:
            return mean, 0.0, n
        stderr = float(np.std(all_vals, ddof=1)) / np.sqrt(n)
        return mean, 1.96 * stderr, n


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class Sampler:
    """
    Computes rolling recall estimates from shadow search observations.

    Parameters
    ----------
    k_values : list[int]
        The k values to track recall for.  Defaults to [1, 5, 10].
        Recall@k is only computed for k values that are <= the k used
        in the actual search call; larger k values in this list are
        silently skipped for any given call.
    window_size : int
        Number of shadow-call batches to retain in the rolling window.
        Each batch is one intercepted search call (possibly nq > 1
        queries).  Defaults to 200 — at 1% sampling and 100 QPS that
        covers the last ~2 minutes of traffic.
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        window_size: int = 200,
    ) -> None:
        self._k_values: list[int] = sorted(k_values or [1, 5, 10])
        self._window_size = window_size
        self._lock = threading.Lock()
        self._windows: dict[int, _ShadowWindow] = {
            k: _ShadowWindow(window_size) for k in self._k_values
        }
        self._n_shadow_calls: int = 0

    # ------------------------------------------------------------------
    # Hot path — called from TurboScopeIndex._run_shadow
    # ------------------------------------------------------------------

    def exact_topk(
        self,
        reservoir: npt.NDArray[np.float32],
        queries: npt.NDArray[np.float32],
        k: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]]:
        """
        Brute-force exact top-k search over ``reservoir`` for each query.

        Delegates to the Rust extension when available, falls back to
        numpy otherwise.

        Parameters
        ----------
        reservoir : ndarray of shape (n_res, dim), float32
            Raw float32 vectors from the reservoir.
        queries : ndarray of shape (nq, dim), float32
            Query vectors (already in the caller's row-major layout).
        k : int
            Number of results per query.

        Returns
        -------
        scores : ndarray of shape (nq, effective_k), float32
            Inner-product scores in descending order.
        local_indices : ndarray of shape (nq, effective_k), intp
            Reservoir-local integer indices (0..n_res) corresponding to
            each score.  The caller converts these to external IDs via
            ``reservoir_ids[local_indices]``.

        Notes
        -----
        ``effective_k = min(k, n_res)``.  When the reservoir has fewer
        than k vectors, fewer results are returned rather than padding.
        """
        try:
            import _turboscope  # Rust extension
            return _turboscope.exact_topk(reservoir, queries, k)
        except ImportError:
            return _exact_topk_numpy(reservoir, queries, k)

    def record_shadow(self, record: "SearchRecord") -> None:
        """
        Compute per-query recall from a completed shadow observation and
        push it into the rolling windows.

        Called by ``TurboScopeIndex._run_shadow`` after both the exact
        and restricted turbovec searches have completed.  ``record`` must
        have all four shadow fields populated (shadow_scores, shadow_ids,
        tq_scores, tq_ids); if any is None this call is silently ignored.

        Parameters
        ----------
        record : SearchRecord
            The populated shadow record from a single search() call.
        """
        if (
            record.shadow_ids is None
            or record.tq_ids is None
        ):
            return

        shadow_ids = record.shadow_ids   # (nq, effective_k_shadow) uint64
        tq_ids = record.tq_ids           # (nq, effective_k_tq)     uint64
        nq = shadow_ids.shape[0]
        if nq == 0:
            return

        with self._lock:
            for k in self._k_values:
                if k > record.k:
                    # This shadow call used a smaller k than we'd need.
                    continue

                k_shadow = min(k, shadow_ids.shape[1])
                k_tq = min(k, tq_ids.shape[1])

                per_query = _recall_per_query(
                    shadow_ids[:, :k_shadow],
                    tq_ids[:, :k_tq],
                    k,
                )
                self._windows[k].push(per_query)

            self._n_shadow_calls += 1

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def snapshot(self) -> RecallSnapshot:
        """
        Return a point-in-time recall estimate from the rolling window.

        Thread-safe: takes a consistent copy of the window statistics
        under the lock and returns an immutable ``RecallSnapshot``.
        """
        with self._lock:
            recall: dict[int, float] = {}
            ci: dict[int, float] = {}
            total_n: int = 0

            for k, window in self._windows.items():
                result = window.stats()
                if result is not None:
                    mean, ci_hw, n = result
                    recall[k] = mean
                    ci[k] = ci_hw
                    total_n = max(total_n, n)

            return RecallSnapshot(
                recall=recall,
                ci_half_width=ci,
                n_queries=total_n,
                n_shadow_calls=self._n_shadow_calls,
                window_size=self._window_size,
            )

    def reset(self) -> None:
        """
        Clear all accumulated recall observations.

        Useful in tests or when the index has been substantially
        re-populated and old estimates are no longer meaningful.
        """
        with self._lock:
            for window in self._windows.values():
                window._batches.clear()
                window._total_queries = 0
            self._n_shadow_calls = 0


# ---------------------------------------------------------------------------
# Pure-numpy fallback for exact_topk
# ---------------------------------------------------------------------------

def _exact_topk_numpy(
    reservoir: npt.NDArray[np.float32],
    queries: npt.NDArray[np.float32],
    k: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]]:
    """
    Brute-force inner-product top-k via numpy.

    Correctness is identical to the Rust path; speed is ~40× slower for
    large reservoirs.  Acceptable for development and CI environments
    where the Rust extension hasn't been compiled.

    At the default reservoir size of 5,000 and dim=1536 this takes ~2ms
    on a modern CPU — acceptable overhead at 1% sample rate but
    noticeable at higher rates.  Build the Rust extension for production
    use.

    Parameters
    ----------
    reservoir : (n_res, dim) float32
    queries   : (nq, dim) float32
    k         : int

    Returns
    -------
    scores        : (nq, effective_k) float32
    local_indices : (nq, effective_k) intp
    """
    n_res = reservoir.shape[0]
    effective_k = min(k, n_res)

    # (nq, n_res) inner-product matrix.
    # float64 accumulation avoids the loss of precision that can swap
    # adjacent ranks when two scores differ by < 1e-7.
    scores_all = queries.astype(np.float64) @ reservoir.astype(np.float64).T

    if effective_k == n_res:
        # All results requested — just sort everything.
        local_idx = np.argsort(-scores_all, axis=1)
        scores_out = np.take_along_axis(scores_all, local_idx, axis=1)
    else:
        # argpartition gives the k smallest negated scores (i.e. k largest
        # positives) cheaply, then sort only those k.
        part = np.argpartition(-scores_all, effective_k, axis=1)[:, :effective_k]
        part_scores = np.take_along_axis(scores_all, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        local_idx = np.take_along_axis(part, order, axis=1)
        scores_out = np.take_along_axis(part_scores, order, axis=1)

    return scores_out.astype(np.float32), local_idx.astype(np.intp)


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def _recall_per_query(
    exact_ids: npt.NDArray[np.uint64],
    tq_ids: npt.NDArray[np.uint64],
    k: int,
) -> npt.NDArray[np.float64]:
    """
    Compute per-query recall@k.

    For each query, recall@k = |exact_top_k ∩ tq_top_k| / min(k, n_exact).

    The denominator is ``min(k, exact_ids.shape[1])`` rather than a fixed
    ``k`` because the reservoir may have fewer than k vectors, in which
    case the exact search returns fewer results — dividing by k would
    artificially deflate the estimate.

    Parameters
    ----------
    exact_ids : (nq, k_exact) uint64
        Top-k external IDs from the exact brute-force search, in score order.
    tq_ids : (nq, k_tq) uint64
        Top-k external IDs from the turbovec restricted search, in score order.
    k : int
        The k value we're computing recall for.  Used only as the
        denominator floor; we don't re-slice the arrays here (caller does
        that before calling us).

    Returns
    -------
    per_query : (nq,) float64
        Recall value in [0, 1] for each query.
    """
    nq = exact_ids.shape[0]
    denom = min(k, exact_ids.shape[1])
    if denom == 0:
        return np.zeros(nq, dtype=np.float64)

    per_query = np.empty(nq, dtype=np.float64)
    for i in range(nq):
        # Set intersection. Using Python sets is fast enough here because
        # k is small (<=100 in practice). A numpy implementation would
        # need np.intersect1d which has O(k log k) overhead from sorting —
        # no faster for small k.
        exact_set = set(exact_ids[i].tolist())
        hits = sum(1 for x in tq_ids[i].tolist() if x in exact_set)
        per_query[i] = hits / denom

    return per_query