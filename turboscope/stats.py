"""
turboscope/stats.py

StatsCollector: records every search() call and exposes a combined
snapshot of latency, throughput, filtering efficiency, recall, and drift.

Design
------
Every search() call produces a SearchRecord (index.py). StatsCollector
consumes it via record() and stores lightweight per-call summaries in a
fixed-size ring buffer. snapshot() reads the buffer under a lock and
returns an immutable Snapshot with all derived metrics.

Recall and drift are not owned here — they live in Sampler and
DriftDetector. snapshot() accepts both as optional arguments and pulls
their current snapshots in to produce a unified Snapshot. This keeps
StatsCollector constructable with no dependencies while giving callers a
single object for all observability data.

The convenience method on TurboScopeIndex (index.snapshot()) wires this
up automatically:

    snap = index.snapshot()
    print(snap.p99_ms, snap.recall_at(10), snap.drift_severity)

Metrics
-------
Latency (per-call wall-clock time):
    p50_ms, p95_ms, p99_ms, mean_ms

Throughput:
    qps           — queries per second over the window duration
    calls_per_s   — search() calls per second over the window duration

Volume:
    total_queries — total queries processed since construction or reset
    total_calls   — total search() calls since construction or reset
    window_calls  — calls currently in the ring buffer

Filtering:
    blocks_skipped_per_query — mean SIMD blocks short-circuited by mask
                               per query. 0.0 when allowlist is never used.
                               Higher means more selective filtering.
    allowlist_rate           — fraction of calls that used an allowlist

Recall (from Sampler.snapshot(), -1.0 if not yet available):
    recall_at_1, recall_at_5, recall_at_10
    recall_ci                — dict[int, float] half-width of 95% CI

Drift (from DriftDetector.snapshot(), -1.0 if baseline not ready):
    drift_severity
    drift_ks_stat
    drift_rel_var_shift
    recalibrate_suggested

Thread safety
-------------
record() and snapshot() are both safe to call from multiple threads.
All mutable state is protected by a single lock. snapshot() copies the
relevant window slice before computing percentiles so the lock is held
for O(window_size) copy time only, not for the numpy computation.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # Type-only imports. These names appear solely in string annotations
    # ("SearchRecord", "Sampler", "DriftDetector"), so importing them here
    # gives type checkers the symbols without creating a runtime circular
    # import (index.py imports stats.py for its SearchRecord type hint).
    from turboscope.index import SearchRecord
    from turboscope.sampler import Sampler
    from turboscope.drift import DriftDetector


# ---------------------------------------------------------------------------
# Snapshot — the unified read-side object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Snapshot:
    """
    Point-in-time view of all observability signals.

    All float fields that require accumulated data return -1.0 when
    insufficient data is available (fewer than 2 calls in the window for
    latency percentiles, no shadow samples for recall, baseline not
    established for drift).

    Attributes
    ----------
    p50_ms, p95_ms, p99_ms, mean_ms : float
        Latency percentiles and mean over the ring-buffer window, in
        milliseconds per search() call.
    qps : float
        Queries per second averaged over the window duration.
    calls_per_s : float
        search() calls per second over the window duration.
    total_queries : int
        Cumulative queries processed since construction or last reset().
    total_calls : int
        Cumulative search() calls since construction or last reset().
    window_calls : int
        Number of calls currently in the ring buffer.
    blocks_skipped_per_query : float
        Mean SIMD blocks short-circuited by mask filtering, per query.
        Requires turbovec to expose blocks_skipped_by_mask(); 0.0 otherwise.
    allowlist_rate : float
        Fraction of calls in the window that used an allowlist.
    recall_at_1, recall_at_5, recall_at_10 : float
        Estimated recall@k from shadow sampling. -1.0 when unavailable.
    recall_ci : dict[int, float]
        95% CI half-width for each tracked k. Empty when unavailable.
    drift_severity : float
        Combined distribution shift score in [0, 1]. -1.0 when the
        baseline has not yet been established.
    drift_ks_stat : float
        KS statistic on L2 norm distributions. -1.0 when unavailable.
    drift_rel_var_shift : float
        Relative variance shift across embedding dimensions. -1.0 when
        unavailable.
    recalibrate_suggested : bool
        True when drift_severity > RECALIBRATE_THRESHOLD.
    """
    # Latency
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    # Throughput
    qps: float
    calls_per_s: float
    # Volume
    total_queries: int
    total_calls: int
    window_calls: int
    # Filtering
    blocks_skipped_per_query: float
    allowlist_rate: float
    # Recall
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_ci: dict[int, float]
    # Drift
    drift_severity: float
    drift_ks_stat: float
    drift_rel_var_shift: float
    recalibrate_suggested: bool

    def recall_at(self, k: int) -> float:
        """Return recall@k. -1.0 if not tracked or not yet available."""
        if k == 1:
            return self.recall_at_1
        if k == 5:
            return self.recall_at_5
        if k == 10:
            return self.recall_at_10
        return -1.0

    def is_ready(self) -> bool:
        """True once at least 2 calls have been recorded in the window."""
        return self.window_calls >= 2

    def summary(self) -> str:
        """Single-line human-readable summary."""
        if not self.is_ready():
            return f"Snapshot(no data — {self.total_calls} calls total)"

        recall_str = (
            f"R@10={self.recall_at_10:.3f}" if self.recall_at_10 >= 0
            else "R@10=n/a"
        )
        drift_str = (
            f"drift={self.drift_severity:.3f}"
            + (" ⚠" if self.recalibrate_suggested else "")
            if self.drift_severity >= 0
            else "drift=n/a"
        )
        return (
            f"Snapshot("
            f"p50={self.p50_ms:.2f}ms "
            f"p99={self.p99_ms:.2f}ms "
            f"qps={self.qps:.1f} "
            f"{recall_str} "
            f"{drift_str})"
        )

    def to_dict(self) -> dict:
        """
        Flat dict of all fields. Useful for JSON export or Prometheus
        scraping. recall_ci is inlined as recall_ci_k1, recall_ci_k5, etc.
        """
        d = {
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "qps": self.qps,
            "calls_per_s": self.calls_per_s,
            "total_queries": self.total_queries,
            "total_calls": self.total_calls,
            "window_calls": self.window_calls,
            "blocks_skipped_per_query": self.blocks_skipped_per_query,
            "allowlist_rate": self.allowlist_rate,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "drift_severity": self.drift_severity,
            "drift_ks_stat": self.drift_ks_stat,
            "drift_rel_var_shift": self.drift_rel_var_shift,
            "recalibrate_suggested": self.recalibrate_suggested,
        }
        for k, v in self.recall_ci.items():
            d[f"recall_ci_k{k}"] = v
        return d


# ---------------------------------------------------------------------------
# _CallRecord — one entry in the ring buffer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _CallRecord:
    """
    Minimal per-call storage. One instance per search() call in the buffer.
    Stored as Python scalars to keep the ring buffer's memory footprint
    predictable (~80 bytes each × 10_000 slots ≈ 800 KB).
    """
    timestamp: float    # time.monotonic() at call end
    latency_ms: float   # wall-clock ms for the turbovec search
    n_queries: int      # nq for this call
    blocks_skipped: int # delta of BLOCKS_SKIPPED_BY_MASK
    had_allowlist: bool


# ---------------------------------------------------------------------------
# StatsCollector
# ---------------------------------------------------------------------------

class StatsCollector:
    """
    Records search() calls and exposes a unified observability snapshot.

    Parameters
    ----------
    window_size : int
        Maximum number of search() call records to retain. Older entries
        are evicted when the buffer is full. Defaults to 10_000.
        At a typical ~80 bytes per record, 10_000 records costs ~800 KB.
    """

    def __init__(self, window_size: int = 10_000) -> None:
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")

        self._window_size = window_size
        self._lock = threading.Lock()

        # Ring buffer implemented as a fixed-size list with a write pointer.
        # This avoids deque overhead for the latency percentile path:
        # numpy percentile on a contiguous float64 array is faster than
        # iterating a deque.
        self._buf: list[_CallRecord | None] = [None] * window_size
        self._head: int = 0         # next write position (mod window_size)
        self._n_stored: int = 0     # calls currently in buffer (<=window_size)

        # Cumulative counters — never reset with the ring buffer.
        self._total_queries: int = 0
        self._total_calls: int = 0

    # ------------------------------------------------------------------
    # Write path — called from TurboScopeIndex.search()
    # ------------------------------------------------------------------

    def record(self, rec: "SearchRecord") -> None:
        """
        Record one search() call.

        Parameters
        ----------
        rec : SearchRecord
            The record produced by TurboScopeIndex.search().
        """
        entry = _CallRecord(
            timestamp=time.monotonic(),
            latency_ms=rec.latency_s * 1_000.0,
            n_queries=rec.n_queries,
            blocks_skipped=rec.blocks_skipped,
            had_allowlist=rec.had_allowlist,
        )

        with self._lock:
            self._buf[self._head] = entry
            self._head = (self._head + 1) % self._window_size
            if self._n_stored < self._window_size:
                self._n_stored += 1
            self._total_queries += rec.n_queries
            self._total_calls += 1

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def snapshot(
        self,
        sampler: "Sampler | None" = None,
        drift: "DriftDetector | None" = None,
    ) -> Snapshot:
        """
        Return a unified point-in-time observability snapshot.

        Parameters
        ----------
        sampler : Sampler or None
            When provided, recall estimates from its rolling window are
            included. When None, recall fields are -1.0.
        drift : DriftDetector or None
            When provided, drift signals are included. When None, drift
            fields are -1.0.

        Returns
        -------
        Snapshot
            Immutable snapshot of all current metrics.
        """
        with self._lock:
            n = self._n_stored
            total_queries = self._total_queries
            total_calls = self._total_calls

            if n == 0:
                return _empty_snapshot(
                    total_queries, total_calls, self._window_size,
                    sampler=sampler, drift=drift,
                )

            # Extract the live portion of the ring buffer in chronological
            # order. The ring is a circular buffer; the oldest entry is at
            # _head when the buffer is full, or index 0 otherwise.
            if n < self._window_size:
                records: list[_CallRecord] = [
                    self._buf[i]  # type: ignore[misc]
                    for i in range(n)
                ]
            else:
                # Buffer is full: oldest is at _head, newest at _head-1.
                records = [
                    self._buf[(self._head + i) % self._window_size]  # type: ignore[misc]
                    for i in range(self._window_size)
                ]

        # All computation below is outside the lock.
        latency_arr = np.array(
            [r.latency_ms for r in records], dtype=np.float64
        )
        n_queries_arr = np.array(
            [r.n_queries for r in records], dtype=np.int64
        )
        blocks_arr = np.array(
            [r.blocks_skipped for r in records], dtype=np.int64
        )
        allowlist_arr = np.array(
            [r.had_allowlist for r in records], dtype=bool
        )
        timestamps = np.array(
            [r.timestamp for r in records], dtype=np.float64
        )

        # Latency percentiles — per-call, not per-query.
        p50, p95, p99 = np.percentile(latency_arr, [50, 95, 99])
        mean_ms = float(np.mean(latency_arr))

        # Throughput over the window duration.
        window_duration_s = float(timestamps[-1] - timestamps[0])
        total_window_queries = int(np.sum(n_queries_arr))
        if window_duration_s > 0.0:
            qps = total_window_queries / window_duration_s
            calls_per_s = len(records) / window_duration_s
        else:
            # All calls happened within the same clock tick — use 0 to
            # avoid division by zero rather than returning inf.
            qps = 0.0
            calls_per_s = 0.0

        # Filtering efficiency.
        total_blocks = int(np.sum(blocks_arr))
        if total_window_queries > 0:
            blocks_skipped_per_query = total_blocks / total_window_queries
        else:
            blocks_skipped_per_query = 0.0
        allowlist_rate = float(np.mean(allowlist_arr))

        # Recall from sampler.
        recall_at_1 = recall_at_5 = recall_at_10 = -1.0
        recall_ci: dict[int, float] = {}
        if sampler is not None:
            rs = sampler.snapshot()
            recall_at_1 = rs.recall_at(1)
            recall_at_5 = rs.recall_at(5)
            recall_at_10 = rs.recall_at(10)
            recall_ci = dict(rs.ci_half_width)

        # Drift from detector.
        drift_severity = drift_ks = drift_rv = -1.0
        recalibrate = False
        if drift is not None:
            ds = drift.snapshot()
            drift_severity = ds.severity
            drift_ks = ds.ks_stat
            drift_rv = ds.rel_var_shift
            recalibrate = ds.recalibrate_suggested

        return Snapshot(
            p50_ms=float(p50),
            p95_ms=float(p95),
            p99_ms=float(p99),
            mean_ms=mean_ms,
            qps=qps,
            calls_per_s=calls_per_s,
            total_queries=total_queries,
            total_calls=total_calls,
            window_calls=n,
            blocks_skipped_per_query=blocks_skipped_per_query,
            allowlist_rate=allowlist_rate,
            recall_at_1=recall_at_1,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            recall_ci=recall_ci,
            drift_severity=drift_severity,
            drift_ks_stat=drift_ks,
            drift_rel_var_shift=drift_rv,
            recalibrate_suggested=recalibrate,
        )

    def reset(self) -> None:
        """
        Clear the ring buffer.

        Cumulative counters (total_queries, total_calls) are also reset.
        Does not affect the Sampler or DriftDetector, which have their
        own reset() methods.
        """
        with self._lock:
            self._buf = [None] * self._window_size
            self._head = 0
            self._n_stored = 0
            self._total_queries = 0
            self._total_calls = 0

    @property
    def total_calls(self) -> int:
        """Cumulative search() calls since construction or last reset."""
        with self._lock:
            return self._total_calls

    @property
    def total_queries(self) -> int:
        """Cumulative queries processed since construction or last reset."""
        with self._lock:
            return self._total_queries

    @property
    def window_size(self) -> int:
        """Maximum number of call records retained."""
        return self._window_size


# ---------------------------------------------------------------------------
# TurboScopeIndex.snapshot() wiring — added here to avoid index.py import
# ---------------------------------------------------------------------------

def attach_snapshot_method(index_cls: type) -> None:
    """
    Attach a .snapshot() convenience method to TurboScopeIndex.

    Called once by turboscope/__init__.py after both modules are imported.
    This avoids a circular import: index.py can't import stats.py at
    module level because stats.py type-hints SearchRecord from index.py.

    The attached method calls:
        self._stats.snapshot(sampler=self._sampler, drift=self._drift)

    which returns a fully populated Snapshot with recall and drift signals
    included whenever the respective components are initialised.
    """
    def snapshot(self) -> "Snapshot":
        """
        Combined observability snapshot: latency, throughput, recall,
        and drift in one object.

        Equivalent to:
            index.stats.snapshot(sampler=..., drift=...)

        but with the sampler and drift detector wired in automatically.

        Returns
        -------
        Snapshot
            Immutable snapshot of all current metrics.
        """
        return self._get_stats().snapshot(
            sampler=self._get_sampler(),
            drift=self._get_drift(),
        )

    index_cls.snapshot = snapshot  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_snapshot(
    total_queries: int,
    total_calls: int,
    window_size: int,
    sampler: "Sampler | None" = None,
    drift: "DriftDetector | None" = None,
) -> Snapshot:
    """
    Return a no-data Snapshot for the latency/throughput fields.

    Drift and recall are still queried when their components are provided —
    both accumulate through add_with_ids(), not search(), so they may have
    real data even when the search ring buffer is empty.
    """
    recall_at_1 = recall_at_5 = recall_at_10 = -1.0
    recall_ci: dict[int, float] = {}
    if sampler is not None:
        rs = sampler.snapshot()
        recall_at_1 = rs.recall_at(1)
        recall_at_5 = rs.recall_at(5)
        recall_at_10 = rs.recall_at(10)
        recall_ci = dict(rs.ci_half_width)

    drift_severity = drift_ks = drift_rv = -1.0
    recalibrate = False
    if drift is not None:
        ds = drift.snapshot()
        drift_severity = ds.severity
        drift_ks = ds.ks_stat
        drift_rv = ds.rel_var_shift
        recalibrate = ds.recalibrate_suggested

    return Snapshot(
        p50_ms=-1.0,
        p95_ms=-1.0,
        p99_ms=-1.0,
        mean_ms=-1.0,
        qps=0.0,
        calls_per_s=0.0,
        total_queries=total_queries,
        total_calls=total_calls,
        window_calls=0,
        blocks_skipped_per_query=0.0,
        allowlist_rate=0.0,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        recall_ci=recall_ci,
        drift_severity=drift_severity,
        drift_ks_stat=drift_ks,
        drift_rel_var_shift=drift_rv,
        recalibrate_suggested=recalibrate,
    )