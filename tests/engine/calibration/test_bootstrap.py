"""Tests for backtest bootstrapping of calibration data."""

import json
import os
import tempfile

import numpy as np
import pytest

from services.engine.calibration.bootstrap import (
    build_calibration_maps_from_backtest,
    calibrate_and_renormalise,
    derive_market_outcomes,
    load_backtest_predictions,
    RENORM_GROUPS,
)


@pytest.fixture
def backtest_json_path():
    """Path to the actual backtest report."""
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "backtests", "20260923T164903Z_5620197.json",
    )
    path = os.path.normpath(path)
    if not os.path.exists(path):
        pytest.skip("Backtest JSON not available")
    return path


@pytest.fixture
def small_backtest_path():
    """Create a minimal backtest JSON for unit tests."""
    data = {
        "predictions": [
            {
                "result": "H",
                "home_goals": 2,
                "away_goals": 1,
                "model_home": 0.55,
                "model_draw": 0.25,
                "model_away": 0.20,
                "lambda_home": 1.6,
                "lambda_away": 1.1,
                "bookmaker_home": 0.50,
                "bookmaker_draw": 0.27,
                "bookmaker_away": 0.23,
            },
            {
                "result": "D",
                "home_goals": 1,
                "away_goals": 1,
                "model_home": 0.40,
                "model_draw": 0.30,
                "model_away": 0.30,
                "lambda_home": 1.3,
                "lambda_away": 1.2,
                "bookmaker_home": 0.42,
                "bookmaker_draw": 0.28,
                "bookmaker_away": 0.30,
            },
            {
                "result": "A",
                "home_goals": 0,
                "away_goals": 2,
                "model_home": 0.30,
                "model_draw": 0.25,
                "model_away": 0.45,
                "lambda_home": 0.9,
                "lambda_away": 1.8,
                "bookmaker_home": 0.28,
                "bookmaker_draw": 0.26,
                "bookmaker_away": 0.46,
            },
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        path = f.name

    yield path
    os.unlink(path)


def test_loads_backtest_json(backtest_json_path):
    """Loading the actual backtest produces 760 predictions."""
    predictions = load_backtest_predictions(backtest_json_path)
    assert len(predictions) == 760
    assert all("model_home" in p for p in predictions)


def test_derives_1x2_outcomes(small_backtest_path):
    """Deriving 1X2 outcomes produces 3 binary observations per prediction."""
    predictions = load_backtest_predictions(small_backtest_path)
    outcomes = derive_market_outcomes(predictions)
    assert len(outcomes["match_result_home"]) == 3
    assert len(outcomes["match_result_draw"]) == 3
    assert len(outcomes["match_result_away"]) == 3
    # Check correct observed values
    home_obs = [o["observed"] for o in outcomes["match_result_home"]]
    assert home_obs == [1, 0, 0]  # H, D, A → 1, 0, 0


def test_derives_over_under_from_lambdas(small_backtest_path):
    """Over/Under 2.5 outcomes derived from lambdas match actual goals."""
    predictions = load_backtest_predictions(small_backtest_path)
    outcomes = derive_market_outcomes(predictions)
    ou_obs = [o["observed"] for o in outcomes["over_under_2.5_over"]]
    # Match 1: 2+1=3 goals → over (1)
    # Match 2: 1+1=2 goals → under (0)
    # Match 3: 0+2=2 goals → under (0)
    assert ou_obs == [1, 0, 0]


def test_derives_btts_from_lambdas(small_backtest_path):
    """BTTS outcomes derived from lambdas match actual goal pattern."""
    predictions = load_backtest_predictions(small_backtest_path)
    outcomes = derive_market_outcomes(predictions)
    btts_obs = [o["observed"] for o in outcomes["btts_yes"]]
    # Match 1: 2-1 → both scored (1)
    # Match 2: 1-1 → both scored (1)
    # Match 3: 0-2 → home didn't score (0)
    assert btts_obs == [1, 1, 0]


def test_full_pipeline_produces_calibration_maps(small_backtest_path):
    """Full pipeline produces calibration maps with correct structure."""
    maps = build_calibration_maps_from_backtest(small_backtest_path, n_bins=5)
    assert len(maps) > 0
    # Should have match_result_home, match_result_draw, etc.
    assert "match_result_home" in maps
    for key, bins in maps.items():
        assert len(bins) == 5
        assert bins[0]["bin_lower"] == pytest.approx(0.0)
        assert bins[-1]["bin_upper"] == pytest.approx(1.0)
        assert all("sample_size" in b for b in bins)


def test_calibrated_1x2_sums_to_one(small_backtest_path):
    """After per-prediction renormalisation, calibrated 1X2 sums to 1."""
    maps = build_calibration_maps_from_backtest(small_backtest_path, n_bins=5)
    predictions = load_backtest_predictions(small_backtest_path)
    group = ["match_result_home", "match_result_draw", "match_result_away"]
    for p in predictions:
        raw_probs = {
            "match_result_home": p["model_home"],
            "match_result_draw": p["model_draw"],
            "match_result_away": p["model_away"],
        }
        calibrated = calibrate_and_renormalise(maps, raw_probs, group)
        total = sum(calibrated.values())
        assert total == pytest.approx(1.0, abs=1e-10)


def test_calibrated_complements_sum_to_one(small_backtest_path):
    """Complementary pairs (over/under, btts yes/no) sum to 1 after renorm."""
    maps = build_calibration_maps_from_backtest(small_backtest_path, n_bins=5)
    # Test with a sample prediction
    raw_probs = {"over_under_2.5_over": 0.55, "over_under_2.5_under": 0.45}
    group = ["over_under_2.5_over", "over_under_2.5_under"]
    calibrated = calibrate_and_renormalise(maps, raw_probs, group)
    total = sum(calibrated.values())
    assert total == pytest.approx(1.0, abs=1e-10)
