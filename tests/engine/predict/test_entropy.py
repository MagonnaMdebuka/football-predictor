"""Tests for confidence bucket classification."""

import pytest

from services.engine.predict.entropy import compute_confidence


def test_uniform_distribution_is_low():
    """Uniform 1/3 each → maximum entropy → 'low' confidence."""
    assert compute_confidence(1 / 3, 1 / 3, 1 / 3) == "low"


def test_strong_favourite_is_very_high():
    """Dominant favourite → low entropy → 'very_high' confidence."""
    assert compute_confidence(0.8, 0.1, 0.1) == "very_high"


def test_moderate_favourite_is_high():
    """Clear favourite but not dominant → 'high'."""
    # entropy(0.65, 0.175, 0.175) ≈ 1.28 → high [1.0, 1.35)
    assert compute_confidence(0.65, 0.175, 0.175) == "high"


def test_competitive_match_is_medium():
    """Fairly balanced → 'medium'."""
    # entropy(0.55, 0.25, 0.20) ≈ 1.44 → medium [1.35, 1.50)
    assert compute_confidence(0.55, 0.25, 0.20) == "medium"


def test_probabilities_must_sum_to_one():
    """Probabilities that do not sum to ~1.0 raise ValueError."""
    with pytest.raises(ValueError, match="sum to 1"):
        compute_confidence(0.5, 0.5, 0.5)
