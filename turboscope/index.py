
"""
turboscope/index.py
 
TurboScopeIndex: a drop-in wrapper around turbovec. IdMapIndex that
intercepts add_with_ids() and search() to maintain a reservoir of raw
float32 vectors for shadow-recall sampling and drift detection.
 
Usage
-----
    from turboscope import TurboScopeIndex
    index = TurboScopeIndex(dim=1536, bit_width=4)
    index.add_with_ids(vectors, ids)
    scores, ids = index.search(query, k=10)
 
    # At any point:
    snap = index.stats.snapshot()
    print(snap.recall_at_10, snap.drift_severity)
 
The wrapper is intentionally thin. It owns three things the inner
IdMapIndex does not:
    1. A reservoir of raw float32 vectors (for the sampler/drift modules).
    2. A StatsCollector that records every search call.
    3. Optional handles to a Sampler and DriftDetector, injected at
       construction so the wrapper doesn't hard-import them (keeps this
       module testable without the Rust extension built).
"""
 
from __future__ import annotations
 
import time
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
 
import numpy as np
import numpy.typing as npt
 
from turbovec import IdMapIndex

# Imported eagerly: drift.py has no turboscope imports, so there is no
# circular-import risk, and DriftDetector is the one sub-component the wrapper
# always constructs by default (see _get_drift). Sampler/StatsCollector stay
# TYPE_CHECKING-only + lazily imported because they are optional.
from turboscope.drift import DriftDetector

if TYPE_CHECKING:
    # Avoid circular imports at runtime; sampler imports this module.
    from turboscope.sampler import Sampler
    from turboscope.stats import StatsCollector
 
 
# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------
 
class Reservoir:
    """
    Fixed-size uniform random sample of raw float32 vectors.
 
    Uses Vitter's Algorithm R so that after seeing N vectors the reservoir
    holds a uniform random subset of size min(N, capacity). The reservoir
    is thread-safe for concurrent adds and reads: adds are serialised by a
    lock; reads (via .vectors) take a snapshot copy so the caller isn't
    racing with an ongoing add.
 
    Parameters
    ----------
    capacity : int
        Maximum number of vectors to retain. 5_000 is the default used by
        TurboScopeIndex — large enough for stable recall estimates, small
        enough that even at dim=3072 the memory cost is ~60 MB.
    seed : int
        RNG seed for reproducibility. Defaults to 0.
    """
 
    def __init__(self, capacity: int = 5_000, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
 
        # Allocated lazily on the first add so we don't need dim up front.
        self._buf: npt.NDArray[np.float32] | None = None
        # Parallel ID buffer: stores the external uint64 ID for each slot in
        # _buf. Required so _run_shadow can pass reservoir_ids as an allowlist
        # to the turbovec restricted search and map local indices to external
        # IDs for recall intersection. Allocated lazily alongside _buf.
        self._ids: npt.NDArray[np.uint64] | None = None
        self._n_seen: int = 0      # total vectors seen (not just stored)
        self._n_stored: int = 0    # vectors currently in the buffer
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def add(self, vectors: npt.NDArray[np.float32], ids: npt.NDArray[np.uint64]) -> None:
        """
        Add a batch of row-vectors to the reservoir.
 
        Parameters
        ----------
        vectors : ndarray of shape (n, dim), dtype float32
            The raw (pre-quantization) vectors. Must be C-contiguous.
        ids : ndarray of shape (n,), dtype uint64
            External IDs matching each row of ``vectors``. Stored alongside
            the vectors so shadow recall can map local indices to external IDs.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D (n, dim), got shape {vectors.shape}"
            )
        n, dim = vectors.shape
        if n == 0:
            return
 
        with self._lock:
            # Lazy allocation: on the first add, size the buffer.
            if self._buf is None:
                self._buf = np.empty(
                    (self._capacity, dim), dtype=np.float32
                )
                # Allocate the parallel ID buffer at the same time.
                self._ids = np.empty(self._capacity, dtype=np.uint64)
            elif self._buf.shape[1] != dim:
                raise ValueError(
                    f"reservoir dim {self._buf.shape[1]} != incoming dim {dim}"
                )
 
            self._reservoir_add(vectors, ids, n, dim)
 
    @property
    def vectors(self) -> npt.NDArray[np.float32] | None:
        """
        A snapshot copy of the current reservoir contents, or None if
        no vectors have been added yet.
 
        Returns shape (n_stored, dim). The copy is taken under the lock
        so the caller always sees a consistent slice.
        """
        with self._lock:
            if self._buf is None or self._n_stored == 0:
                return None
            return self._buf[: self._n_stored].copy()
 
    @property
    def vectors_and_ids(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint64]] | None:
        """
        Snapshot copy of (vectors, ids) for the current reservoir contents,
        or None if no vectors have been added yet.
 
        Both arrays are shape (n_stored, dim) and (n_stored,) respectively.
        The copy is taken under the lock for consistency.
        """
        with self._lock:
            if self._buf is None or self._n_stored == 0 or self._ids is None:
                return None
            return (
                self._buf[: self._n_stored].copy(),
                self._ids[: self._n_stored].copy(),
            )
 
    @property
    def n_seen(self) -> int:
        """Total number of vectors offered to the reservoir (not just stored)."""
        with self._lock:
            return self._n_seen
 
    @property
    def n_stored(self) -> int:
        """Number of vectors currently in the reservoir."""
        with self._lock:
            return self._n_stored
 
    @property
    def dim(self) -> int | None:
        """Dimensionality, or None before the first add."""
        with self._lock:
            return None if self._buf is None else self._buf.shape[1]
 
    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
 
    def _reservoir_add(
        self,
        vectors: npt.NDArray[np.float32],
        ids: npt.NDArray[np.uint64],
        n: int,
        dim: int,
    ) -> None:
        """
        Algorithm R (Vitter 1985). Called with self._lock held.
 
        For each incoming vector at overall position self._n_seen + i:
            - If the reservoir isn't full yet, append it.
            - Otherwise, pick a random slot j in [0, position]; if j <
              capacity, replace buf[j] with this vector.
        Both the vector and its external ID are written to the same slot
        so _ids[slot] always corresponds to _buf[slot].
        """
        buf = self._buf  # not None; guaranteed by caller
        id_buf = self._ids  # not None; guaranteed by caller
        cap = self._capacity
 
        for i in range(n):
            pos = self._n_seen + i        # 0-based overall index of this vector
            if self._n_stored < cap:
                # Reservoir not full: always accept.
                buf[self._n_stored] = vectors[i]
                id_buf[self._n_stored] = ids[i]  # keep ID in sync with vector slot
                self._n_stored += 1
            else:
                # Reservoir full: accept with probability cap / (pos + 1).
                j = int(self._rng.integers(0, pos + 1))
                if j < cap:
                    buf[j] = vectors[i]
                    id_buf[j] = ids[i]  # replace both vector and ID atomically
                # else: discard this vector.
 
        self._n_seen += n
 
 
# ---------------------------------------------------------------------------
# SearchRecord — one intercepted search call
# ---------------------------------------------------------------------------
 
@dataclass(slots=True)
class SearchRecord:
    """
    Metadata captured for a single search() call.
 
    Passed to StatsCollector.record() and, when sampled, to
    Sampler.record_shadow().
    """
    n_queries: int
    k: int
    latency_s: float           # wall-clock seconds for the turbovec search
    n_results: int             # total results returned (nq * effective_k)
    blocks_skipped: int        # delta of BLOCKS_SKIPPED_BY_MASK over this call
    had_allowlist: bool
    # Filled in by the sampler when this call was shadow-sampled:
    shadow_scores: npt.NDArray[np.float32] | None = field(default=None)
    shadow_ids: npt.NDArray[np.uint64] | None = field(default=None)
    tq_scores: npt.NDArray[np.float32] | None = field(default=None)
    tq_ids: npt.NDArray[np.uint64] | None = field(default=None)
 
 
# ---------------------------------------------------------------------------
# TurboScopeIndex
# ---------------------------------------------------------------------------
 
class TurboScopeIndex:
    """
    Observability wrapper around :class:`turbovec.IdMapIndex`.
 
    Mirrors the complete public API of IdMapIndex so it can be used as a
    drop-in replacement. Adds:
 
    - A :class:`Reservoir` of raw float32 vectors for shadow sampling.
    - A :class:`~turboscope.stats.StatsCollector` recording every search.
    - Optional :class:`~turboscope.sampler.Sampler` and
      :class:`~turboscope.drift.DriftDetector` injected at construction.
 
    Parameters
    ----------
    dim : int or None
        Vector dimensionality. ``None`` for lazy construction — the index
        commits to a dim on the first ``add_with_ids`` call.
    bit_width : int
        Quantization width, 2 or 4.
    reservoir_size : int
        Number of raw float32 vectors to retain for shadow sampling.
        Defaults to 5_000.
    sample_rate : float
        Fraction of search calls on which to run shadow recall estimation.
        0.01 (1%) is the default. Set to 0.0 to disable sampling entirely.
    reservoir_seed : int
        RNG seed for reservoir sampling.
    sampler : Sampler or None
        Injected sampler. When None and sample_rate > 0, a default
        :class:`~turboscope.sampler.Sampler` is created on first use.
    drift_detector : DriftDetector or None
        Injected drift detector. When None, a default
        :class:`~turboscope.drift.DriftDetector` is created on first use.
    stats_collector : StatsCollector or None
        Injected stats collector. When None, a default
        :class:`~turboscope.stats.StatsCollector` is created.
    index : IdMapIndex or None
        Pre-built inner index. When given, ``dim`` and ``bit_width`` are
        ignored. Useful for loading a serialized index and wrapping it.
    """
 
    def __init__(
        self,
        dim: int | None = None,
        bit_width: int = 4,
        *,
        reservoir_size: int = 5_000,
        sample_rate: float = 0.01,
        reservoir_seed: int = 0,
        sampler: "Sampler | None" = None,
        drift_detector: "DriftDetector | None" = None,
        stats_collector: "StatsCollector | None" = None,
        index: IdMapIndex | None = None,
    ) -> None:
        if index is not None:
            self._index = index
        else:
            self._index = IdMapIndex(dim=dim, bit_width=bit_width)
 
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError(
                f"sample_rate must be in [0, 1], got {sample_rate}"
            )
 
        self._reservoir = Reservoir(capacity=reservoir_size, seed=reservoir_seed)
        self._sample_rate = sample_rate
        self._rng = np.random.default_rng(reservoir_seed + 1)
 
        # Lazy imports to avoid circular dependency and to allow the wrapper
        # to be imported even when the Rust extension isn't built yet.
        self._sampler: "Sampler | None" = sampler
        self._drift: "DriftDetector | None" = drift_detector
        self._stats: "StatsCollector | None" = stats_collector
 
        # Tracks the global BLOCKS_SKIPPED_BY_MASK counter between calls.
        self._last_blocks_skipped: int = self._read_blocks_skipped()
 
    # ------------------------------------------------------------------
    # turbovec.IdMapIndex public API — pass-through with interception
    # ------------------------------------------------------------------
 
    def add_with_ids(
        self,
        vectors: npt.NDArray[np.float32],
        ids: npt.NDArray[np.uint64],
    ) -> None:
        """
        Add vectors with stable external uint64 IDs.
 
        Forwards to :meth:`turbovec.IdMapIndex.add_with_ids` and then
        feeds the raw float32 vectors into the reservoir.
 
        Raises
        ------
        ValueError
            If ``vectors`` and ``ids`` have mismatched lengths, if any id
            is already present, or if the dim doesn't match the index.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
 
        # Let turbovec validate first — if it raises, we don't update the
        # reservoir (keeps the two stores consistent).
        self._index.add_with_ids(vectors, ids)
 
        # Only feed the reservoir after a successful add.
        # Pass ids so the reservoir can map local indices to external IDs
        # when building the shadow-recall allowlist.
        self._reservoir.add(vectors, ids)
 
        # Feed the raw vectors to the drift detector. We go through the lazy
        # getter (not a bare `if self._drift is not None`) so that, per the
        # constructor contract, a default DriftDetector is created on first
        # use when none was injected. observe() — not the on_add() no-op stub —
        # is what records norms and per-dim variance; without this call the
        # baseline is never established and drift_severity stays -1.0 forever.
        self._get_drift().observe(vectors)
 
    def search(
        self,
        queries: npt.NDArray[np.float32],
        k: int,
        *,
        allowlist: npt.NDArray[np.uint64] | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint64]]:
        """
        Top-k search. Mirrors :meth:`turbovec.IdMapIndex.search`.
 
        When ``allowlist`` is provided, results are restricted to those
        external IDs. Passes through to the inner index unchanged — the
        allowlist filtering happens inside the turbovec SIMD kernel.
 
        Returns
        -------
        scores : ndarray of shape (nq, effective_k), float32
        ids : ndarray of shape (nq, effective_k), uint64
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
 
        # Snapshot the block-skip counter before the search so we can
        # compute the delta afterwards.
        before_skipped = self._read_blocks_skipped()
 
        t0 = time.perf_counter()
        if allowlist is not None:
            allowlist = np.ascontiguousarray(allowlist, dtype=np.uint64)
            scores, ids = self._index.search(queries, k, allowlist=allowlist)
        else:
            scores, ids = self._index.search(queries, k)
        latency = time.perf_counter() - t0
 
        after_skipped = self._read_blocks_skipped()
        blocks_delta = after_skipped - before_skipped
 
        record = SearchRecord(
            n_queries=queries.shape[0],
            k=k,
            latency_s=latency,
            n_results=scores.size,
            blocks_skipped=blocks_delta,
            had_allowlist=allowlist is not None,
        )
 
        # Shadow sampling: fire on a random fraction of calls.
        if self._sample_rate > 0.0 and self._rng.random() < self._sample_rate:
            self._run_shadow(queries, k, scores, ids, record)
 
        # Record into the stats collector.
        self._get_stats().record(record)
 
        return scores, ids
 
    def remove(self, id: int) -> bool:
        """
        Remove the vector with the given external ID.
 
        Returns True if the id was present, False otherwise. Forwarded
        directly to the inner index — the reservoir is not shrunk (it's a
        sample, not a mirror).
        """
        return self._index.remove(id)
 
    def contains(self, id: int) -> bool:
        """Return True if the given external ID is in the index."""
        return self._index.contains(id)
 
    def prepare(self) -> None:
        """
        Warm up the search caches (rotation matrix, Lloyd-Max centroids,
        SIMD-blocked code layout). Forwards to the inner index.
        """
        self._index.prepare()
 
    def write(self, path: str) -> None:
        """
        Serialize the index to a ``.tvim`` file.
 
        Only the turbovec index is written — the reservoir is intentionally
        not persisted. On load, the reservoir starts empty and fills as new
        vectors are added.
        """
        self._index.write(path)
 
    @classmethod
    def load(cls, path: str, **kwargs) -> "TurboScopeIndex":
        """
        Load a ``.tvim`` file and wrap it in a :class:`TurboScopeIndex`.
 
        All keyword arguments (``reservoir_size``, ``sample_rate``, etc.)
        are forwarded to the constructor.
 
        Example
        -------
        ::
 
            index = TurboScopeIndex.load("my_index.tvim", sample_rate=0.05)
        """
        inner = IdMapIndex.load(path)
        return cls(index=inner, **kwargs)
 
    # ------------------------------------------------------------------
    # Properties that mirror IdMapIndex's read-only attributes
    # ------------------------------------------------------------------
 
    @property
    def dim(self) -> int | None:
        """Dimensionality, or None before the first add."""
        return self._index.dim
 
    @property
    def bit_width(self) -> int:
        """Quantization width (2 or 4)."""
        return self._index.bit_width
 
    def __len__(self) -> int:
        return len(self._index)
 
    def __contains__(self, id: int) -> bool:
        return self._index.contains(id)
 
    def __repr__(self) -> str:
        return (
            f"TurboScopeIndex("
            f"dim={self.dim}, "
            f"bit_width={self.bit_width}, "
            f"n_vectors={len(self)}, "
            f"reservoir={self._reservoir.n_stored}/{self._reservoir._capacity})"
        )
 
    # ------------------------------------------------------------------
    # the turboscope-specific accessors
    # ------------------------------------------------------------------
 
    @property
    def reservoir(self) -> Reservoir:
        """The raw-vector reservoir used by the sampler and drift detector."""
        return self._reservoir
 
    @property
    def stats(self) -> "StatsCollector":
        """
        The :class:`~turboscope.stats.StatsCollector` for this index.
 
        Created lazily on first access if not injected at construction.
        """
        return self._get_stats()
 
    @property
    def sample_rate(self) -> float:
        """Fraction of searches that trigger shadow recall estimation."""
        return self._sample_rate
 
    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"sample_rate must be in [0, 1], got {value}")
        self._sample_rate = value
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _run_shadow(
        self,
        queries: npt.NDArray[np.float32],
        k: int,
        tq_scores: npt.NDArray[np.float32],
        tq_ids: npt.NDArray[np.uint64],
        record: SearchRecord,
    ) -> None:
        sampler = self._get_sampler()
        if sampler is None:
            return
 
        snap = self._reservoir.vectors_and_ids
        if snap is None:
            return
 
        reservoir_vectors, reservoir_ids = snap
        if len(reservoir_vectors) == 0:
            return
 
        try:
            # Exact brute-force top-k over raw reservoir vectors — ground truth
            # for this candidate set.  Returns local 0-based indices.
            shadow_scores, local_indices = sampler.exact_topk(
                reservoir_vectors, queries, k
            )
            # Map reservoir-local indices to external uint64 IDs so the
            # recall intersection in record_shadow works correctly.
            shadow_ids = reservoir_ids[local_indices].astype(np.uint64)
 
            # Turbovec search restricted to the same reservoir IDs — this is
            # the approximate result for the identical candidate set, so the
            # recall estimate is unbiased (both searches see the same pool).
            _, tq_ids_res = self._index.search(
                queries, k, allowlist=reservoir_ids
            )
 
            record.shadow_scores = shadow_scores
            record.shadow_ids = shadow_ids
            record.tq_scores = tq_scores   # full-index scores kept for reference
            record.tq_ids = tq_ids_res     # reservoir-restricted turbovec IDs
 
            sampler.record_shadow(record)
        except Exception:
            pass
 
    def _get_sampler(self) -> "Sampler | None":
        """Lazily construct the default sampler."""
        if self._sampler is None and self._sample_rate > 0.0:
            try:
                from turboscope.sampler import Sampler
                self._sampler = Sampler()
            except ImportError:
                return None
        return self._sampler
 
    def _get_drift(self) -> "DriftDetector":
        """
        Lazily construct the default drift detector.

        Unlike the sampler, the drift detector is always created on first
        use (there is no rate that disables it): drift tracking has no
        per-call sampling cost — it only summarizes each add() batch — so
        it runs unconditionally once any vectors are added.
        """
        if self._drift is None:
            self._drift = DriftDetector()
        return self._drift

    def _get_stats(self) -> "StatsCollector":
        """Lazily construct the default stats collector."""
        if self._stats is None:
            from turboscope.stats import StatsCollector
            self._stats = StatsCollector()
        return self._stats
 
    def _read_blocks_skipped(self) -> int:
        """
        Read the turbovec global block-skip counter.
 
        turbovec exposes this as ``turbovec.search.blocks_skipped_by_mask()``.
        We import lazily and fall back to 0 silently so the wrapper stays
        usable even against older turbovec builds that don't expose it.
        """
        try:
            from turbovec.search import blocks_skipped_by_mask
            return int(blocks_skipped_by_mask())
        except ImportError:
            return 0