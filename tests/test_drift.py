"""
Tests for turboscope.drift.

These exercise the DriftDetector and its two signal kernels directly. They
depend only on numpy (drift.py does not import turbovec), so they run even
when the Rust extension and turbovec are not available.

A small ``min_baseline_size`` is used throughout to keep the baseline cheap
to establish; the production default is 1,000.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboscope.drift import (
    DriftDetector,
    DriftSnapshot,
    _ks_2samp,
    _relative_var_shift,
)


def _gaussian(n: int, dim: int, *, loc: float = 0.0, scale: float = 1.0,
              seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(loc=loc, scale=scale, size=(n, dim))).astype(np.float32)


# ---------------------------------------------------------------------------
# Signal kernels
# ---------------------------------------------------------------------------

def test_ks_identical_is_zero():
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert _ks_2samp(a, a.copy()) == 0.0


def test_ks_disjoint_is_one():
    a = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    b = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    assert _ks_2samp(a, b) == pytest.approx(1.0)


def test_ks_in_unit_range():
    a = _gaussian(500, 1, seed=1).ravel()
    b = _gaussian(500, 1, loc=0.5, seed=2).ravel()
    ks = _ks_2samp(a, b)
    assert 0.0 <= ks <= 1.0
    # A half-sigma shift on 500 samples is clearly detectable.
    assert ks > 0.1


def test_relative_var_shift_identical_is_zero():
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert _relative_var_shift(v, v.copy()) == 0.0


def test_relative_var_shift_degenerate_baseline_is_zero():
    base = np.zeros(8, dtype=np.float64)
    curr = np.ones(8, dtype=np.float64)
    # median baseline ~0 -> guarded to 0.0 rather than dividing by ~0.
    assert _relative_var_shift(base, curr) == 0.0


def test_relative_var_shift_positive_on_change():
    base = np.full(8, 1.0, dtype=np.float64)
    curr = np.full(8, 3.0, dtype=np.float64)
    # median(|3-1|)/median(1) = 2.0
    assert _relative_var_shift(base, curr) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Baseline lifecycle
# ---------------------------------------------------------------------------

def test_snapshot_before_baseline_is_not_ready():
    d = DriftDetector(min_baseline_size=200)
    d.observe(_gaussian(50, 16, seed=0))
    snap = d.snapshot()
    assert isinstance(snap, DriftSnapshot)
    assert snap.baseline_ready is False
    assert snap.severity == -1.0
    assert snap.ks_stat == -1.0
    assert snap.n_baseline == 50


def test_baseline_becomes_ready_across_batches():
    d = DriftDetector(min_baseline_size=200)
    # Three batches summing past the threshold.
    d.observe(_gaussian(80, 16, seed=1))
    assert d.snapshot().baseline_ready is False
    d.observe(_gaussian(80, 16, seed=2))
    assert d.snapshot().baseline_ready is False
    d.observe(_gaussian(80, 16, seed=3))
    snap = d.snapshot()
    assert snap.baseline_ready is True
    assert snap.n_baseline >= 200


def test_baseline_ready_no_window_is_zero_severity():
    d = DriftDetector(min_baseline_size=100)
    d.observe(_gaussian(100, 16, seed=4))   # exactly fills the baseline
    snap = d.snapshot()
    assert snap.baseline_ready is True
    # Baseline established, nothing observed since -> honest 0.0, not -1.0.
    assert snap.severity == 0.0
    assert snap.n_current_window == 0


# ---------------------------------------------------------------------------
# Severity behaviour
# ---------------------------------------------------------------------------

def test_same_distribution_low_severity():
    d = DriftDetector(min_baseline_size=400)
    # Baseline.
    d.observe(_gaussian(400, 32, seed=10))
    # Post-baseline window from the *same* distribution.
    for s in range(11, 16):
        d.observe(_gaussian(200, 32, seed=s))
    snap = d.snapshot()
    assert snap.baseline_ready is True
    assert snap.n_current_window > 0
    assert 0.0 <= snap.severity < 0.3
    assert snap.recalibrate_suggested is False


def test_shifted_distribution_high_severity_suggests_recalibration():
    d = DriftDetector(min_baseline_size=400)
    # Baseline: unit-scale, centred at origin.
    d.observe(_gaussian(400, 32, loc=0.0, scale=1.0, seed=20))
    # Window: a very different distribution — larger scale and offset, which
    # moves both the norm distribution (KS) and per-dim variance (rel_var).
    for s in range(21, 26):
        d.observe(_gaussian(200, 32, loc=2.0, scale=3.0, seed=s))
    snap = d.snapshot()
    assert snap.baseline_ready is True
    assert snap.ks_stat > 0.5
    assert snap.severity > 0.3
    assert snap.recalibrate_suggested is True


def test_observe_ignores_empty_and_1d():
    d = DriftDetector(min_baseline_size=50)
    d.observe(np.empty((0, 16), dtype=np.float32))   # empty batch
    d.observe(np.zeros(16, dtype=np.float32))         # 1-D, not (n, dim)
    assert d.snapshot().n_baseline == 0


# ---------------------------------------------------------------------------
# Reset and validation
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    d = DriftDetector(min_baseline_size=100)
    d.observe(_gaussian(150, 16, seed=30))
    assert d.snapshot().baseline_ready is True
    d.reset()
    snap = d.snapshot()
    assert snap.baseline_ready is False
    assert snap.n_baseline == 0
    assert snap.n_current_window == 0


@pytest.mark.parametrize("kwargs", [
    {"min_baseline_size": 0},
    {"window_calls": 0},
    {"recalibrate_threshold": 0.0},
    {"recalibrate_threshold": 1.0},
])
def test_invalid_constructor_args_raise(kwargs):
    with pytest.raises(ValueError):
        DriftDetector(**kwargs)
