"""
turboscope — observability wrapper for turbovec.

Live recall estimation, distribution-shift detection, and latency monitoring
for quantized vector search, exposed as a drop-in wrapper around
``turbovec.IdMapIndex``.

Quick start
-----------
::

    from turboscope import TurboScopeIndex

    index = TurboScopeIndex(dim=1536, bit_width=4)
    index.add_with_ids(vectors, ids)
    scores, ids = index.search(query, k=10)

    snap = index.snapshot()            # unified Snapshot (latency/recall/drift)
    print(snap.recall_at(10), snap.drift_severity)

    from turboscope import Dashboard
    Dashboard(index).print_once()      # one-shot terminal render

Importing this package requires the hard dependencies declared in
``pyproject.toml`` (``turbovec``, ``numpy``, ``rich``); the optional Rust
extension ``turboscope._turboscope`` only accelerates the shadow sampler and
is not required for correctness.
"""

from __future__ import annotations

from turboscope.index import Reservoir, SearchRecord, TurboScopeIndex
from turboscope.sampler import RecallSnapshot, Sampler
from turboscope.drift import DriftDetector, DriftSnapshot
from turboscope.stats import Snapshot, StatsCollector, attach_snapshot_method
from turboscope.dashboard import Dashboard, print_snapshot

# Wire the unified .snapshot() convenience method onto TurboScopeIndex. This
# lives in stats.py (so index.py needn't import stats.py at module level and
# risk a circular import) and must be attached exactly once, at import time.
attach_snapshot_method(TurboScopeIndex)

__version__ = "0.1.0"

__all__ = [
    "TurboScopeIndex",
    "Reservoir",
    "SearchRecord",
    "Sampler",
    "RecallSnapshot",
    "DriftDetector",
    "DriftSnapshot",
    "StatsCollector",
    "Snapshot",
    "Dashboard",
    "print_snapshot",
    "__version__",
]
