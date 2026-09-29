"""Tests for scoring metrics."""

import numpy as np
import pytest

from services.engine.calibration.accuracy import (
    brier_score,
    compute_accuracy_metrics,
    hit_rate,
    rps_score,
)


def test_brier_perfect_is_zero():
    """Brier score of perfect predictions is 0."""
    predicted = np.array([1.0, 0.0, 1.0, 0.0])
    observed = np.array([1, 0, 1, 0])
    assert brier_score(predicted, observed) == pytest.approx(0.0)


def test_brier_uniform_on_binary():
    """Brier score of 0.5 predictions on balanced binary = 0.25."""
    predicted = np.array([0.5, 0.5, 0.5, 0.5])
    observed = np.array([1, 0, 1, 0])
    assert brier_score(predicted, observed) == pytest.approx(0.25)


def test_rps_perfect_prediction():
    """RPS of a perfect 1X2 prediction is 0."""
    # Predict home win with certainty, and home wins
    probs = np.array([1.0, 0.0, 0.0])
    assert rps_score(probs, outcome_idx=0) == pytest.approx(0.0)


def test_hit_rate_computed_correctly():
    """Hit rate should be fraction of correct predictions."""
    predicted = np.array([0, 1, 2, 1, 0])
    actual = np.array([0, 1, 2, 0, 0])
    assert hit_rate(predicted, actual) == pytest.approx(0.8)
