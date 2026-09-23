"""Tests for backtest metrics: RPS, log loss, Brier, hit rate."""

from __future__ import annotations

import numpy as np
import pytest

from services.engine.backtest.metrics import (
    brier_score,
    compute_metric_set,
    hit_rate,
    log_loss,
    ranked_probability_score,
)
from services.engine.backtest.types import (
    BacktestConfig,
    BookmakerOddsCols,
    GateDetail,
    MatchPrediction,
    MetricSet,
)

# ---------------------------------------------------------------------------
# Types — frozen dataclass tests
# ---------------------------------------------------------------------------

class TestTypes:
    """Verify dataclass behaviour and defaults."""

    def test_backtest_config_defaults(self):
        cfg = BacktestConfig(
            held_out_seasons=("2024-25",),
            training_start_season="2019-20",
        )
        assert cfg.xi == 0.0065
        assert cfg.refit_step == "per_date"
        assert cfg.seed == 42
        assert cfg.max_goals == 11
        assert cfg.rho_bounds == (-0.5, 0.5)

    def test_backtest_config_frozen(self):
        cfg = BacktestConfig(
            held_out_seasons=("2024-25",),
            training_start_season="2019-20",
        )
        with pytest.raises(AttributeError):
            cfg.xi = 0.01  # type: ignore[misc]

    def test_bookmaker_odds_cols_defaults(self):
        cols = BookmakerOddsCols()
        assert cols.primary_home == "PSCH"
        assert cols.fallback_away == "AvgCA"

    def test_metric_set_frozen(self):
        ms = MetricSet(
            rps=0.2, log_loss=1.0, brier_home=0.2,
            brier_draw=0.3, brier_away=0.25, n_matches=100, hit_rate=0.53,
        )
        assert ms.n_matches == 100
        with pytest.raises(AttributeError):
            ms.rps = 0.1  # type: ignore[misc]

    def test_match_prediction_nullable_bookmaker(self):
        pred = MatchPrediction(
            date="2024-08-17", season="2024-25",
            home_team="A", away_team="B",
            home_goals=2, away_goals=1, result="H", matchday=1,
            model_home=0.5, model_draw=0.25, model_away=0.25,
            uniform_home=1 / 3, uniform_draw=1 / 3, uniform_away=1 / 3,
            base_rate_home=0.45, base_rate_draw=0.27, base_rate_away=0.28,
            indep_poisson_home=0.48, indep_poisson_draw=0.26, indep_poisson_away=0.26,
            bookmaker_home=None, bookmaker_draw=None, bookmaker_away=None,
            n_training_matches=500,
        )
        assert pred.bookmaker_home is None

    def test_gate_detail_fields(self):
        gd = GateDetail(name="rps_vs_base_rate", passed=True, message="0.200 < 0.210")
        assert gd.passed is True


# ---------------------------------------------------------------------------
# RPS tests
# ---------------------------------------------------------------------------

class TestRankedProbabilityScore:
    """RPS for 1X2 outcomes."""

    def test_perfect_prediction(self):
        """RPS = 0 when predicted probabilities match actuals exactly."""
        p_home = np.array([1.0, 0.0, 0.0])
        p_draw = np.array([0.0, 1.0, 0.0])
        p_away = np.array([0.0, 0.0, 1.0])
        actual = np.array(["H", "D", "A"])
        assert ranked_probability_score(p_home, p_draw, p_away, actual) == pytest.approx(0.0)

    def test_worst_prediction(self):
        """RPS is maximised when all mass is on the wrong extreme outcome."""
        # Predict full away but actual is home
        p_home = np.array([0.0])
        p_draw = np.array([0.0])
        p_away = np.array([1.0])
        actual = np.array(["H"])
        rps = ranked_probability_score(p_home, p_draw, p_away, actual)
        assert rps == pytest.approx(1.0)

    def test_uniform_rps(self):
        """Uniform 1/3 predictions should give known RPS values."""
        p = np.array([1 / 3])
        actual = np.array(["H"])
        rps = ranked_probability_score(p, p, p, actual)
        # cum_p1 = 1/3, cum_o1 = 1 => (1/3 - 1)^2 = 4/9
        # cum_p2 = 2/3, cum_o2 = 1 => (2/3 - 1)^2 = 1/9
        # RPS = 0.5 * (4/9 + 1/9) = 5/18
        assert rps == pytest.approx(5 / 18, abs=1e-10)

    def test_empty_returns_zero(self):
        empty = np.array([], dtype=np.float64)
        actual = np.array([], dtype=np.str_)
        assert ranked_probability_score(empty, empty, empty, actual) == 0.0

    def test_symmetric_home_away(self):
        """RPS for predicting full home when result is away should equal
        RPS for predicting full away when result is home."""
        rps_ha = ranked_probability_score(
            np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array(["A"]),
        )
        rps_ah = ranked_probability_score(
            np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array(["H"]),
        )
        assert rps_ha == pytest.approx(rps_ah)


# ---------------------------------------------------------------------------
# Log loss tests
# ---------------------------------------------------------------------------

class TestLogLoss:
    """Categorical log loss for 1X2."""

    def test_perfect_prediction(self):
        p_home = np.array([1.0 - 1e-15])
        p_draw = np.array([1e-15 / 2])
        p_away = np.array([1e-15 / 2])
        actual = np.array(["H"])
        assert log_loss(p_home, p_draw, p_away, actual) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_log_loss(self):
        """Uniform 1/3 should give log(3) ≈ 1.0986."""
        p = np.array([1 / 3])
        actual = np.array(["H"])
        assert log_loss(p, p, p, actual) == pytest.approx(np.log(3), abs=1e-10)

    def test_clipping_prevents_infinity(self):
        """Zero probability should not produce infinity due to clipping."""
        p_home = np.array([0.0])
        p_draw = np.array([0.0])
        p_away = np.array([1.0])
        actual = np.array(["H"])
        result = log_loss(p_home, p_draw, p_away, actual)
        assert np.isfinite(result)

    def test_empty_returns_zero(self):
        empty = np.array([], dtype=np.float64)
        actual = np.array([], dtype=np.str_)
        assert log_loss(empty, empty, empty, actual) == 0.0


# ---------------------------------------------------------------------------
# Brier score tests
# ---------------------------------------------------------------------------

class TestBrierScore:
    """Brier score for single binary outcome."""

    def test_perfect_prediction(self):
        p = np.array([1.0, 0.0])
        actual = np.array([1.0, 0.0])
        assert brier_score(p, actual) == pytest.approx(0.0)

    def test_worst_prediction(self):
        p = np.array([0.0])
        actual = np.array([1.0])
        assert brier_score(p, actual) == pytest.approx(1.0)

    def test_uniform_brier(self):
        """P=0.5 for event that happened => Brier = 0.25."""
        p = np.array([0.5])
        actual = np.array([1.0])
        assert brier_score(p, actual) == pytest.approx(0.25)

    def test_empty_returns_zero(self):
        assert brier_score(np.array([]), np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Hit rate tests
# ---------------------------------------------------------------------------

class TestHitRate:
    """Hit rate: fraction of correct predictions."""

    def test_all_correct(self):
        p_home = np.array([0.6, 0.1, 0.2])
        p_draw = np.array([0.2, 0.2, 0.3])
        p_away = np.array([0.2, 0.7, 0.5])
        actual = np.array(["H", "A", "A"])
        assert hit_rate(p_home, p_draw, p_away, actual) == pytest.approx(1.0)

    def test_none_correct(self):
        p_home = np.array([0.6])
        p_draw = np.array([0.2])
        p_away = np.array([0.2])
        actual = np.array(["A"])
        assert hit_rate(p_home, p_draw, p_away, actual) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        empty = np.array([], dtype=np.float64)
        actual = np.array([], dtype=np.str_)
        assert hit_rate(empty, empty, empty, actual) == 0.0


# ---------------------------------------------------------------------------
# compute_metric_set
# ---------------------------------------------------------------------------

class TestComputeMetricSet:
    """Integration test for computing all metrics at once."""

    def test_returns_correct_tuple_length(self):
        p = np.array([1 / 3, 1 / 3])
        actual = np.array(["H", "D"])
        result = compute_metric_set(p, p, p, actual)
        assert len(result) == 7

    def test_n_matches_correct(self):
        p = np.array([0.5, 0.3, 0.2])
        actual = np.array(["H", "D", "A"])
        result = compute_metric_set(p, np.array([0.3, 0.4, 0.3]), np.array([0.2, 0.3, 0.5]), actual)
        assert result[5] == 3  # n_matches
