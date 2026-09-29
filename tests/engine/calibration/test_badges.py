"""Tests for quality badge assignment."""

from services.engine.calibration.badges import compute_quality_badge


def test_three_plus_seasons_is_green():
    """3+ seasons of direct data should produce green badge."""
    assert compute_quality_badge(n_seasons=3, has_direct_data=True) == "green"
    assert compute_quality_badge(n_seasons=5, has_direct_data=True) == "green"


def test_one_or_two_seasons_is_amber():
    """1-2 seasons of direct data should produce amber badge."""
    assert compute_quality_badge(n_seasons=1, has_direct_data=True) == "amber"
    assert compute_quality_badge(n_seasons=2, has_direct_data=True) == "amber"


def test_no_data_is_grey():
    """Zero seasons or no direct data should produce grey badge."""
    assert compute_quality_badge(n_seasons=0, has_direct_data=True) == "grey"
    assert compute_quality_badge(n_seasons=5, has_direct_data=False) == "grey"
