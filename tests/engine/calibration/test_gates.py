"""Tests for publication gate checks."""

from services.engine.calibration.gates import check_publication_gates


def _good_reliability():
    return {"calibration_error": 0.03, "mean_calibration_error": 0.02}


def test_all_gates_pass_with_good_data():
    """All 4 gates pass when data meets all thresholds."""
    results = check_publication_gates(
        reliability_data=_good_reliability(),
        n_settled=600,
        beats_baseline=True,
        has_direct_data=True,
    )
    assert all(g["passed"] for g in results)
    assert len(results) == 4


def test_fails_on_insufficient_settled():
    """Gate fails when fewer than 500 predictions have settled."""
    results = check_publication_gates(
        reliability_data=_good_reliability(),
        n_settled=400,
        beats_baseline=True,
        has_direct_data=True,
    )
    settled_gate = next(g for g in results if g["name"] == "min_settled")
    assert not settled_gate["passed"]


def test_fails_on_high_decile_error():
    """Gate fails when max decile error exceeds 5pp."""
    reliability = {"calibration_error": 0.08, "mean_calibration_error": 0.05}
    results = check_publication_gates(
        reliability_data=reliability,
        n_settled=600,
        beats_baseline=True,
        has_direct_data=True,
    )
    error_gate = next(g for g in results if g["name"] == "max_decile_error")
    assert not error_gate["passed"]


def test_fails_when_no_direct_data():
    """Gate fails when calibration is from inferred data only."""
    results = check_publication_gates(
        reliability_data=_good_reliability(),
        n_settled=600,
        beats_baseline=True,
        has_direct_data=False,
    )
    data_gate = next(g for g in results if g["name"] == "direct_data")
    assert not data_gate["passed"]


def test_sparse_bins_skipped_by_min_bin_count():
    """Bins with fewer than min_bin_count observations are skipped for gate."""
    # One bin has 9 observations and huge error, one bin has 100 and small error
    reliability = {
        "calibration_error": 0.15,  # overall max (from sparse bin)
        "mean_calibration_error": 0.05,
        "bins": [
            {"predicted_mean": 0.10, "observed_mean": 0.25, "count": 9},   # 15pp, sparse
            {"predicted_mean": 0.30, "observed_mean": 0.32, "count": 100},  # 2pp, ok
            {"predicted_mean": 0.50, "observed_mean": 0.48, "count": 80},   # 2pp, ok
        ],
    }
    results = check_publication_gates(
        reliability_data=reliability,
        n_settled=600,
        beats_baseline=True,
        has_direct_data=True,
        min_bin_count=30,
    )
    error_gate = next(g for g in results if g["name"] == "max_decile_error")
    # The 15pp sparse bin is skipped; max among eligible is 2pp → passes
    assert error_gate["passed"]
    assert "sparse bins skipped" in error_gate["message"]
