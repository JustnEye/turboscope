"""
Integration tests for turboscope.index.TurboScopeIndex.

These wrap a real ``turbovec.IdMapIndex``, so the whole module is skipped when
turbovec is not installed (it is a hard runtime dependency of index.py, imported
at module top). The Rust extension is optional — the shadow sampler falls back
to numpy — so these run without a compiled ``_turboscope``.

The focus is the *wiring* between the wrapper and its sub-components, which is
exactly where the end-to-end pipeline was broken:

  * drift must actually receive each add() batch (a default DriftDetector is
    created on first use), so the baseline establishes and severity leaves the
    -1.0 "not ready" sentinel;
  * shadow sampling must actually run and feed recall estimates back;
  * ``index.snapshot()`` must exist (attached at package import).
"""

from __future__ import annotations

import numpy as np
import pytest

# Hard dependency: skip the whole module cleanly if turbovec is unavailable.
pytest.importorskip("turbovec")

# Importing from the package (not the submodule) also verifies __init__.py
# exports and that attach_snapshot_method() ran at import time.
from turboscope import DriftDetector, Reservoir, TurboScopeIndex


DIM = 16


def _vectors(n: int, *, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def _ids(start: int, n: int) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.uint64)


def _populated_index(n: int = 300, *, seed: int = 0, **kwargs) -> TurboScopeIndex:
    idx = TurboScopeIndex(dim=DIM, bit_width=4, **kwargs)
    idx.add_with_ids(_vectors(n, seed=seed), _ids(0, n))
    idx.prepare()
    return idx


# ---------------------------------------------------------------------------
# Reservoir (Algorithm R)
# ---------------------------------------------------------------------------

def test_reservoir_fills_then_caps():
    r = Reservoir(capacity=100, seed=0)
    assert r.n_stored == 0 and r.dim is None
    r.add(_vectors(40, seed=1), _ids(0, 40))
    assert r.n_stored == 40 and r.n_seen == 40 and r.dim == DIM
    r.add(_vectors(200, seed=2), _ids(40, 200))
    assert r.n_stored == 100        # capped at capacity
    assert r.n_seen == 240          # but all were seen


def test_reservoir_vectors_and_ids_stay_aligned():
    r = Reservoir(capacity=10, seed=3)
    vecs = _vectors(10, seed=4)
    ids = _ids(1000, 10)
    r.add(vecs, ids)
    snap = r.vectors_and_ids
    assert snap is not None
    out_vecs, out_ids = snap
    assert out_vecs.shape == (10, DIM)
    assert out_ids.shape == (10,)
    # With no eviction (n_seen == capacity), order is preserved.
    np.testing.assert_array_equal(out_ids, ids)


def test_reservoir_rejects_dim_mismatch():
    r = Reservoir(capacity=10)
    r.add(_vectors(5, dim=8), _ids(0, 5))
    with pytest.raises(ValueError):
        r.add(_vectors(5, dim=9), _ids(5, 5))


def test_reservoir_invalid_capacity():
    with pytest.raises(ValueError):
        Reservoir(capacity=0)


# ---------------------------------------------------------------------------
# Pass-through API parity with IdMapIndex
# ---------------------------------------------------------------------------

def test_basic_passthrough():
    idx = _populated_index(50)
    assert len(idx) == 50
    assert idx.dim == DIM
    assert idx.bit_width == 4
    assert idx.contains(0) and (0 in idx)
    assert not idx.contains(10_000)
    assert idx.remove(0) is True
    assert idx.remove(0) is False
    assert len(idx) == 49
    assert "TurboScopeIndex" in repr(idx)


def test_search_returns_expected_shapes():
    idx = _populated_index(200, seed=7)
    scores, ids = idx.search(_vectors(5, seed=99), k=10)
    assert scores.shape[0] == 5 and ids.shape[0] == 5
    assert scores.shape[1] <= 10
    assert scores.shape == ids.shape


def test_search_accepts_1d_query():
    idx = _populated_index(100, seed=8)
    scores, ids = idx.search(_vectors(1, seed=5)[0], k=5)
    assert scores.shape[0] == 1


# ---------------------------------------------------------------------------
# Stats wiring
# ---------------------------------------------------------------------------

def test_snapshot_method_attached_and_counts_calls():
    idx = _populated_index(100, sample_rate=0.0)   # no shadow overhead
    for s in range(5):
        idx.search(_vectors(2, seed=100 + s), k=10)
    # .snapshot() exists only because __init__ ran attach_snapshot_method().
    snap = idx.snapshot()
    assert snap.total_calls == 5
    assert snap.total_queries == 10
    assert snap.window_calls == 5


# ---------------------------------------------------------------------------
# Drift wiring — the regression this package shipped with
# ---------------------------------------------------------------------------

def test_drift_baseline_establishes_through_add():
    # Inject a small-baseline detector so we don't need 1,000 vectors.
    drift = DriftDetector(min_baseline_size=100)
    idx = TurboScopeIndex(dim=DIM, bit_width=4, drift_detector=drift)

    idx.add_with_ids(_vectors(60, seed=1), _ids(0, 60))
    assert idx.snapshot().drift_severity == -1.0   # baseline not ready yet

    # Cross the baseline and add a post-baseline window.
    idx.add_with_ids(_vectors(120, seed=2), _ids(60, 120))
    snap = idx.snapshot()
    # The key assertion: drift is alive. Before the wiring fix this stayed
    # at the -1.0 sentinel forever because observe() was never called.
    assert snap.drift_severity >= 0.0
    assert drift.snapshot().baseline_ready is True


def test_drift_detector_created_lazily_when_not_injected():
    idx = TurboScopeIndex(dim=DIM, bit_width=4)
    assert idx._drift is None
    idx.add_with_ids(_vectors(10, seed=0), _ids(0, 10))
    # First add must have lazily constructed the default detector.
    assert idx._drift is not None
    assert idx._drift.snapshot().n_baseline == 10


# ---------------------------------------------------------------------------
# Shadow recall wiring
# ---------------------------------------------------------------------------

def test_shadow_sampling_populates_recall():
    # sample_rate=1.0 -> every search runs the shadow path.
    idx = _populated_index(300, seed=11, sample_rate=1.0)
    for s in range(8):
        idx.search(_vectors(4, seed=200 + s), k=10)

    snap = idx.snapshot()
    # Recall must have been computed and fed back — not the -1.0 sentinel.
    assert snap.recall_at_10 >= 0.0
    assert 0.0 <= snap.recall_at_10 <= 1.0
    # Querying with vectors drawn from the index should recover them well;
    # over the reservoir sub-corpus recall should be solidly positive.
    assert snap.recall_at_10 > 0.0


def test_no_shadow_when_sample_rate_zero():
    idx = _populated_index(200, seed=12, sample_rate=0.0)
    for s in range(5):
        idx.search(_vectors(3, seed=300 + s), k=10)
    snap = idx.snapshot()
    assert snap.recall_at_10 == -1.0   # never sampled


def test_sample_rate_validation():
    with pytest.raises(ValueError):
        TurboScopeIndex(dim=DIM, sample_rate=1.5)
    idx = _populated_index(20)
    with pytest.raises(ValueError):
        idx.sample_rate = -0.1
