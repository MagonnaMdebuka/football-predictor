"""Tests for count types and Brier score computation."""

import pytest

from services.engine.backtest.count_types import (
    CountPrediction,
    compute_count_brier,
    compute_count_calibration,
)


def _make_count_pred(
    actual_home: int = 5,
    actual_away: int = 4,
    mu_home: float = 5.0,
    mu_away: float = 4.5,
    alpha: float = 0.15,
    model_type: str = "corners",
    match_ou: list[dict] | None = None,
) -> CountPrediction:
    if match_ou is None:
        match_ou = [
            {"line": 9.5, "over": 0.55, "under": 0.45},
            {"line": 10.5, "over": 0.40, "under": 0.60},
        ]
    return CountPrediction(
        date="2024-08-17",
        season="2024-25",
        home_team="Alpha",
        away_team="Bravo",
        model_type=model_type,
        actual_home=actual_home,
        actual_away=actual_away,
        mu_home=mu_home,
        mu_away=mu_away,
        alpha=alpha,
        match_over_under=match_ou,
        home_over_under=[],
        away_over_under=[],
        n_training_matches=100,
    )


class TestComputeCountBrier:
    def test_empty_predictions(self):
        result = compute_count_brier([])
        assert result.n_predictions == 0
        assert result.mean_brier == 0.0

    def test_single_prediction_correct_over(self):
        """When actual > line and p_over = 1, Brier should be 0."""
        pred = _make_count_pred(
            actual_home=6, actual_away=5,  # total = 11 > 9.5
            match_ou=[{"line": 9.5, "over": 1.0, "under": 0.0}],
        )
        result = compute_count_brier([pred])
        assert result.mean_brier == pytest.approx(0.0)

    def test_single_prediction_wrong(self):
        """When actual < line but p_over = 1, Brier should be 1."""
        pred = _make_count_pred(
            actual_home=3, actual_away=3,  # total = 6 < 9.5
            match_ou=[{"line": 9.5, "over": 1.0, "under": 0.0}],
        )
        result = compute_count_brier([pred])
        assert result.mean_brier == pytest.approx(1.0)

    def test_brier_in_valid_range(self):
        """Brier score should be in [0, 1]."""
        preds = [_make_count_pred() for _ in range(5)]
        result = compute_count_brier(preds)
        assert 0.0 <= result.mean_brier <= 1.0

    def test_per_line_brier_keys(self):
        """Per-line Brier dict should have entries for each line."""
        pred = _make_count_pred()
        result = compute_count_brier([pred])
        assert 9.5 in result.per_line_brier
        assert 10.5 in result.per_line_brier

    def test_n_predictions(self):
        preds = [_make_count_pred() for _ in range(7)]
        result = compute_count_brier(preds)
        assert result.n_predictions == 7

    def test_mean_totals(self):
        pred = _make_count_pred(actual_home=5, actual_away=4, mu_home=5.0, mu_away=4.5)
        result = compute_count_brier([pred])
        assert result.mean_predicted_total == pytest.approx(9.5)
        assert result.mean_actual_total == pytest.approx(9.0)


class TestMuTotalInBrier:
    """compute_count_brier uses mu_total when available."""

    def test_mu_total_used_for_predicted_total(self):
        """When mu_total is set, predicted total should use it instead of mu_home + mu_away."""
        pred = _make_count_pred(
            actual_home=5, actual_away=4, mu_home=5.0, mu_away=4.5,
        )
        # Replace with a version that has mu_total set
        pred_with_total = CountPrediction(
            date=pred.date, season=pred.season, home_team=pred.home_team,
            away_team=pred.away_team, model_type=pred.model_type,
            actual_home=pred.actual_home, actual_away=pred.actual_away,
            mu_home=pred.mu_home, mu_away=pred.mu_away, alpha=pred.alpha,
            match_over_under=pred.match_over_under,
            home_over_under=pred.home_over_under,
            away_over_under=pred.away_over_under,
            n_training_matches=pred.n_training_matches,
            mu_total=11.0,  # different from mu_home + mu_away (9.5)
        )
        result = compute_count_brier([pred_with_total])
        assert result.mean_predicted_total == pytest.approx(11.0)

    def test_without_mu_total_uses_sum(self):
        """Without mu_total, predicted total should be mu_home + mu_away."""
        pred = _make_count_pred(
            actual_home=5, actual_away=4, mu_home=5.0, mu_away=4.5,
        )
        result = compute_count_brier([pred])
        assert result.mean_predicted_total == pytest.approx(9.5)


class TestComputeCountCalibration:
    def test_empty_returns_none(self):
        assert compute_count_calibration([]) is None

    def test_bias_computation(self):
        pred = _make_count_pred(actual_home=5, actual_away=4, mu_home=5.5, mu_away=4.5)
        cal = compute_count_calibration([pred])
        assert cal is not None
        assert cal.n_predictions == 1
        assert cal.predicted_mean_total == pytest.approx(10.0)
        assert cal.actual_mean_total == pytest.approx(9.0)
        assert cal.bias == pytest.approx(1.0)

    def test_zero_bias(self):
        pred = _make_count_pred(actual_home=5, actual_away=5, mu_home=5.0, mu_away=5.0)
        cal = compute_count_calibration([pred])
        assert cal is not None
        assert cal.bias == pytest.approx(0.0)
