"""Tests for PAVA isotonic regression."""

import numpy as np
import pytest

from services.engine.calibration.isotonic import (
    fit_calibration_bins,
    isotonic_regression,
)


def test_already_monotone_is_identity():
    """PAVA on already-monotone data returns the same values."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = isotonic_regression(y)
    np.testing.assert_array_almost_equal(result, y)


def test_reversed_data_gives_constant():
    """PAVA on fully reversed data returns constant (overall mean)."""
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    result = isotonic_regression(y)
    expected = np.full(5, 3.0)
    np.testing.assert_array_almost_equal(result, expected)


def test_pava_partial_violation():
    """PAVA on [0.3, 0.1, 0.5] merges first two into [0.2, 0.2, 0.5]."""
    y = np.array([0.3, 0.1, 0.5])
    result = isotonic_regression(y)
    np.testing.assert_array_almost_equal(result, [0.2, 0.2, 0.5])


def test_empty_input():
    """PAVA on empty array returns empty array."""
    result = isotonic_regression(np.array([]))
    assert len(result) == 0


def test_fit_calibration_bins_returns_10_bins():
    """fit_calibration_bins with default n_bins returns 10 bins."""
    rng = np.random.default_rng(42)
    predicted = rng.uniform(0, 1, 200)
    observed = (rng.uniform(0, 1, 200) < predicted).astype(float)
    bins = fit_calibration_bins(predicted, observed)
    assert len(bins) == 10
    for b in bins:
        assert set(b.keys()) == {
            "bin_lower", "bin_upper", "predicted_frequency",
            "observed_frequency", "sample_size",
        }


def test_bins_cover_unit_interval():
    """Bin edges span [0, 1]."""
    rng = np.random.default_rng(42)
    predicted = rng.uniform(0, 1, 100)
    observed = (rng.uniform(0, 1, 100) < predicted).astype(float)
    bins = fit_calibration_bins(predicted, observed)
    assert bins[0]["bin_lower"] == pytest.approx(0.0)
    assert bins[-1]["bin_upper"] == pytest.approx(1.0)


def test_sample_sizes_sum_to_input():
    """Sample sizes across all bins sum to the number of input samples."""
    rng = np.random.default_rng(42)
    n = 150
    predicted = rng.uniform(0, 1, n)
    observed = (rng.uniform(0, 1, n) < predicted).astype(float)
    bins = fit_calibration_bins(predicted, observed)
    assert sum(b["sample_size"] for b in bins) == n
