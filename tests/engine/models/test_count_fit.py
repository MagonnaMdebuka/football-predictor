"""Tests for NB2 count model fitting."""

import numpy as np
import pytest

from services.engine.models.count_fit import CountFitResult, fit_count_model
from services.engine.models.count_params import pack
from tests.engine.models.conftest import (
    COUNT_ALPHA_CORNERS,
    COUNT_GAMMA_CORNERS,
    COUNT_MU_CORNERS,
    COUNT_TEAMS,
)


class TestFitCorners:
    """Fit corners model (no referees)."""

    def test_convergence(self, synthetic_corners_df):
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.converged

    def test_n_matches(self, synthetic_corners_df):
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.n_matches == len(synthetic_corners_df)

    def test_mu_recovery(self, synthetic_corners_df):
        """Fitted mu should be within 0.3 of true value."""
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.params.mu == pytest.approx(COUNT_MU_CORNERS, abs=0.3)

    def test_gamma_recovery(self, synthetic_corners_df):
        """Fitted gamma should be within 0.15 of true value."""
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.params.gamma == pytest.approx(COUNT_GAMMA_CORNERS, abs=0.15)

    def test_alpha_positive(self, synthetic_corners_df):
        """Fitted alpha should be positive (overdispersion detected)."""
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.params.alpha > 0

    def test_alpha_recovery(self, synthetic_corners_df):
        """Fitted alpha should be within 0.3 of true value."""
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.params.alpha == pytest.approx(COUNT_ALPHA_CORNERS, abs=0.3)

    def test_attack_sums_to_zero(self, synthetic_corners_df):
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.params.attack.sum() == pytest.approx(0.0, abs=1e-10)

    def test_defence_sums_to_zero(self, synthetic_corners_df):
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.params.defence.sum() == pytest.approx(0.0, abs=1e-10)

    def test_no_referees(self, synthetic_corners_df):
        """Corners model should have no referee effects."""
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert len(result.params.referees) == 0

    def test_teams_sorted(self, synthetic_corners_df):
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        assert result.params.teams == sorted(COUNT_TEAMS)


class TestFitCards:
    """Fit cards model (with referees)."""

    def test_convergence(self, synthetic_cards_df):
        result = fit_count_model(
            synthetic_cards_df, "home_booking_points", "away_booking_points",
            include_referees=True, min_referee_matches=10,
        )
        assert result.converged

    def test_has_referee_effects(self, synthetic_cards_df):
        """Cards model with referees should have referee effects."""
        result = fit_count_model(
            synthetic_cards_df, "home_booking_points", "away_booking_points",
            include_referees=True, min_referee_matches=10,
        )
        assert len(result.params.referees) >= 2

    def test_referee_effect_sums_to_zero(self, synthetic_cards_df):
        result = fit_count_model(
            synthetic_cards_df, "home_booking_points", "away_booking_points",
            include_referees=True, min_referee_matches=10,
        )
        if len(result.params.referees) > 0:
            assert result.params.referee_effect.sum() == pytest.approx(0.0, abs=1e-10)

    def test_high_threshold_excludes_referees(self, synthetic_cards_df):
        """With min_referee_matches very high, no referee should qualify."""
        result = fit_count_model(
            synthetic_cards_df, "home_booking_points", "away_booking_points",
            include_referees=True, min_referee_matches=9999,
        )
        assert len(result.params.referees) == 0


class TestLowRateCount:
    """Fit model on low-rate counts (e.g. red cards, mean ~0.1)."""

    def test_low_rate_count_converges(self):
        """Model with mean count ~0.1 should converge with the 0.01 mu floor."""
        from scipy.stats import poisson as poisson_dist

        rng = np.random.default_rng(42)
        teams = ["A", "B", "C", "D"]
        rows = []
        for _ in range(6):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    rows.append({
                        "home_team": h,
                        "away_team": a,
                        "home_reds": int(rng.poisson(0.1)),
                        "away_reds": int(rng.poisson(0.08)),
                    })
        import pandas as pd
        df = pd.DataFrame(rows)
        result = fit_count_model(
            df, "home_reds", "away_reds",
            alpha_bounds=(1e-12, 1e-12),  # force near-Poisson
        )
        assert result.converged
        # Mu should be close to log(mean), around -2.3 for mean ~0.1
        assert result.params.mu < 0  # negative log-rate for rare events


class TestWarmStart:
    """Warm-start from previous fit."""

    def test_warm_start_accepted(self, synthetic_corners_df):
        """Warm-start should produce valid results."""
        result1 = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        x0 = pack(result1.params)
        result2 = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
            x0=x0,
        )
        assert result2.converged

    def test_wrong_length_x0_ignored(self, synthetic_corners_df):
        """x0 with wrong length should be silently ignored."""
        x0_bad = np.zeros(3, dtype=np.float64)
        result = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
            x0=x0_bad,
        )
        assert result.converged

    def test_warm_and_cold_agree(self, synthetic_corners_df):
        """Cold-start and warm-start should agree on mu within tolerance."""
        cold = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
        )
        x0 = pack(cold.params)
        warm = fit_count_model(
            synthetic_corners_df, "home_corners", "away_corners",
            x0=x0,
        )
        assert cold.params.mu == pytest.approx(warm.params.mu, abs=1e-3)
        assert cold.params.gamma == pytest.approx(warm.params.gamma, abs=1e-3)
        assert cold.params.alpha == pytest.approx(warm.params.alpha, abs=1e-3)
