"""Tests for reliability diagram data computation."""

import numpy as np
import pytest

from services.engine.calibration.reliability import compute_reliability_data


def test_perfect_calibration_has_zero_error():
    """When predictions match observed frequencies exactly, error is near zero."""
    # Create data where each bin has observed frequency matching prediction
    rng = np.random.default_rng(42)
    n = 5000
    predicted = rng.uniform(0, 1, n)
    # Generate outcomes matching predicted probabilities
    observed = (rng.uniform(0, 1, n) < predicted).astype(float)
    result = compute_reliability_data(predicted, observed)
    # With enough samples the error should be small
    assert result["mean_calibration_error"] < 0.05


def test_extreme_overconfidence_has_high_error():
    """Predicting 0.9 but observing ~50% should produce high error."""
    predicted = np.full(200, 0.9)
    observed = np.zeros(200)
    observed[:100] = 1.0  # 50% observed
    result = compute_reliability_data(predicted, observed)
    assert result["calibration_error"] > 0.3


def test_bin_midpoints_are_correct():
    """Midpoints of 10 equal-width bins on [0,1] are 0.05, 0.15, ..., 0.95."""
    predicted = np.linspace(0.05, 0.95, 100)
    observed = np.zeros(100)
    result = compute_reliability_data(predicted, observed)
    expected_midpoints = [0.05 + 0.1 * i for i in range(10)]
    actual_midpoints = [b["midpoint"] for b in result["bins"]]
    np.testing.assert_array_almost_equal(actual_midpoints, expected_midpoints)


def test_empty_bins_handled():
    """Bins with no data should have count=0 and not cause errors."""
    # All predictions in [0.0, 0.1) — other bins empty
    predicted = np.full(50, 0.05)
    observed = np.zeros(50)
    result = compute_reliability_data(predicted, observed)
    assert len(result["bins"]) == 10
    non_empty = [b for b in result["bins"] if b["count"] > 0]
    assert len(non_empty) == 1
    assert non_empty[0]["count"] == 50


def test_sparse_bins_flagged_insufficient():
    """Bins below min_bin_count are flagged and excluded from error metrics."""
    # 10 observations in 0.8-0.9 bin (sparse), 100 in 0.4-0.5 bin
    predicted = np.concatenate([np.full(10, 0.85), np.full(100, 0.45)])
    observed = np.concatenate([np.ones(10), np.zeros(50), np.ones(50)])
    result = compute_reliability_data(predicted, observed, min_bin_count=30)
    sparse = [b for b in result["bins"] if b.get("insufficient_data")]
    sufficient = [b for b in result["bins"] if b["count"] > 0 and not b.get("insufficient_data")]
    # The 10-obs bin should be flagged
    assert any(b["count"] == 10 for b in sparse)
    # The 100-obs bin should not be flagged
    assert any(b["count"] == 100 for b in sufficient)
    # calibration_error should only reflect the sufficient bin
    assert result["calibration_error"] < 0.1  # 0.45 vs 0.50 = 0.05
