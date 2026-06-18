"""
Tests for turboscope.sampler.

These cover the recall machinery directly: the numpy exact-top-k fallback,
per-query recall, the rolling window, and the Sampler's record/snapshot
cycle. They depend only on numpy (sampler.py does not import turbovec), and
``Sampler.exact_topk`` transparently falls back to numpy when the Rust
extension is not built — so these run anywhere.

record_shadow only reads ``.shadow_ids``, ``.tq_ids`` and ``.k`` off the
record, so a SimpleNamespace stands in for the full SearchRecord (which lives
in index.py and would pull in turbovec).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from turboscope.sampler import (
    RecallSnapshot,
    Sampler,
    _ShadowWindow,
    _exact_topk_numpy,
    _recall_per_query,
)


def _shadow_record(shadow_ids, tq_ids, k):
    return SimpleNamespace(
        shadow_ids=np.asarray(shadow_ids, dtype=np.uint64),
        tq_ids=np.asarray(tq_ids, dtype=np.uint64),
        k=k,
    )


# ---------------------------------------------------------------------------
# Exact top-k (numpy fallback + the public dispatcher)
# ---------------------------------------------------------------------------

def test_exact_topk_numpy_matches_bruteforce():
    rng = np.random.default_rng(0)
    reservoir = rng.normal(size=(50, 8)).astype(np.float32)
    queries = rng.normal(size=(4, 8)).astype(np.float32)
    k = 5

    scores, idx = _exact_topk_numpy(reservoir, queries, k)
    assert scores.shape == (4, k)
    assert idx.shape == (4, k)

    # Reference via full f64 matmul + argsort.
    full = queries.astype(np.float64) @ reservoir.astype(np.float64).T
    for q in range(queries.shape[0]):
        expected = np.argsort(-full[q])[:k]
        np.testing.assert_array_equal(idx[q], expected)
        # Scores must be non-increasing.
        assert np.all(np.diff(scores[q]) <= 1e-6)


def test_exact_topk_k_larger_than_reservoir():
    reservoir = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    scores, idx = _exact_topk_numpy(reservoir, queries, k=10)
    # effective_k == n_res == 3
    assert scores.shape == (1, 3)
    assert idx.shape == (1, 3)
    assert set(idx[0].tolist()) == {0, 1, 2}


def test_sampler_exact_topk_dispatch_matches_numpy():
    # The Rust extension is not built in CI, so exact_topk falls back to numpy.
    rng = np.random.default_rng(1)
    reservoir = rng.normal(size=(30, 6)).astype(np.float32)
    queries = rng.normal(size=(3, 6)).astype(np.float32)
    s1, i1 = Sampler().exact_topk(reservoir, queries, 4)
    s2, i2 = _exact_topk_numpy(reservoir, queries, 4)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(s1, s2, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Per-query recall
# ---------------------------------------------------------------------------

def test_recall_per_query_perfect():
    exact = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
    tq = exact.copy()
    r = _recall_per_query(exact, tq, k=3)
    np.testing.assert_array_equal(r, np.array([1.0, 1.0]))


def test_recall_per_query_partial():
    exact = np.array([[1, 2, 3, 4]], dtype=np.uint64)
    tq = np.array([[1, 2, 99, 98]], dtype=np.uint64)   # 2 of 4 correct
    r = _recall_per_query(exact, tq, k=4)
    assert r[0] == pytest.approx(0.5)


def test_recall_per_query_denominator_uses_available_exact():
    # Only 2 exact results available -> denominator is 2, not k=5.
    exact = np.array([[1, 2]], dtype=np.uint64)
    tq = np.array([[1, 2]], dtype=np.uint64)
    r = _recall_per_query(exact, tq, k=5)
    assert r[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rolling window
# ---------------------------------------------------------------------------

def test_shadow_window_stats_single_value_zero_ci():
    w = _ShadowWindow(max_calls=10)
    w.push(np.array([0.8], dtype=np.float64))
    mean, ci, n = w.stats()
    assert mean == pytest.approx(0.8)
    assert ci == 0.0
    assert n == 1


def test_shadow_window_eviction_keeps_query_count_consistent():
    w = _ShadowWindow(max_calls=2)
    w.push(np.array([1.0, 1.0], dtype=np.float64))   # 2 queries
    w.push(np.array([0.0], dtype=np.float64))         # 1 query
    w.push(np.array([0.5], dtype=np.float64))         # evicts the first batch
    mean, ci, n = w.stats()
    # Only the last two batches remain: [0.0, 0.5]
    assert n == 2
    assert mean == pytest.approx(0.25)
    assert w._total_queries == 2


def test_shadow_window_empty_returns_none():
    assert _ShadowWindow(max_calls=4).stats() is None


# ---------------------------------------------------------------------------
# Sampler record / snapshot cycle
# ---------------------------------------------------------------------------

def test_snapshot_before_any_record_not_ready():
    snap = Sampler().snapshot()
    assert isinstance(snap, RecallSnapshot)
    assert snap.is_ready() is False
    assert snap.recall_at(10) == -1.0
    assert "no data" in str(snap)


def test_record_shadow_perfect_recall():
    s = Sampler(k_values=[1, 5, 10])
    exact = np.tile(np.arange(10, dtype=np.uint64), (3, 1))   # 3 queries, k=10
    rec = _shadow_record(exact, exact.copy(), k=10)
    s.record_shadow(rec)

    snap = s.snapshot()
    assert snap.is_ready()
    assert snap.n_shadow_calls == 1
    assert snap.recall_at(1) == pytest.approx(1.0)
    assert snap.recall_at(5) == pytest.approx(1.0)
    assert snap.recall_at(10) == pytest.approx(1.0)


def test_record_shadow_partial_recall_at_k():
    s = Sampler(k_values=[10])
    exact = np.arange(10, dtype=np.uint64)[None, :]            # one query
    # turbovec got the first 8 right, missed 2.
    tq = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 100, 101]], dtype=np.uint64)
    s.record_shadow(_shadow_record(exact, tq, k=10))
    assert s.snapshot().recall_at(10) == pytest.approx(0.8)


def test_record_shadow_skips_k_larger_than_search_k():
    s = Sampler(k_values=[1, 5, 10])
    exact = np.arange(3, dtype=np.uint64)[None, :]
    # The actual search used k=3, so recall@5 and @10 must be skipped.
    s.record_shadow(_shadow_record(exact, exact.copy(), k=3))
    snap = s.snapshot()
    assert snap.recall_at(1) == pytest.approx(1.0)
    assert snap.recall_at(5) == -1.0
    assert snap.recall_at(10) == -1.0


def test_record_shadow_ignores_missing_fields():
    s = Sampler()
    s.record_shadow(SimpleNamespace(shadow_ids=None, tq_ids=None, k=10))
    assert s.snapshot().is_ready() is False


def test_reset_clears_windows():
    s = Sampler(k_values=[1])
    exact = np.arange(5, dtype=np.uint64)[None, :]
    s.record_shadow(_shadow_record(exact, exact.copy(), k=5))
    assert s.snapshot().is_ready()
    s.reset()
    assert s.snapshot().is_ready() is False
