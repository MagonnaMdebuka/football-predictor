"""Tests for total count NB2 model fitting."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from services.engine.models.total_count_fit import TotalCountFitResult, fit_total_count_model
from services.engine.models.total_count_params import pack_total

# True parameters for synthetic data
TEAMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
TRUE_MU = 2.35  # log-scale: exp(2.35) ~ 10.5 total corners per match
TRUE_HOME_EFFECT = np.array([0.12, 0.08, 0.02, -0.04, -0.08, -0.10])
TRUE_AWAY_EFFECT = np.array([0.10, 0.04, 0.01, -0.03, -0.06, -0.06])
TRUE_ALPHA = 0.12


@pytest.fixture
def synthetic_total_df() -> pd.DataFrame:
    """Generate 180 matches with NB2-distributed total corner counts."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    base_date = np.datetime64("2024-01-01")
    match_num = 0

    for _repeat in range(6):
        for i, home in enumerate(TEAMS):
            for j, away in enumerate(TEAMS):
                if i == j:
                    continue
                mu_total = np.exp(
                    TRUE_MU + TRUE_HOME_EFFECT[i] + TRUE_AWAY_EFFECT[j]
                )
                n_param = 1.0 / TRUE_ALPHA
                p_param = n_param / (n_param + mu_total)

                total = int(nbinom.rvs(n_param, p_param, random_state=rng))
                # Split total into home/away (rough 55/45 split)
                home_share = max(0, min(total, int(rng.binomial(total, 0.55))))
                away_share = total - home_share

                rows.append({
                    "date": base_date + np.timedelta64(match_num * 3, "D"),
                    "home_team": home,
                    "away_team": away,
                    "home_corners": home_share,
                    "away_corners": away_share,
                })
                match_num += 1

    return pd.DataFrame(rows)


class TestFitTotalCount:
    """Fit total corners model."""

    def test_convergence(self, synthetic_total_df):
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        assert result.converged

    def test_n_matches(self, synthetic_total_df):
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        assert result.n_matches == len(synthetic_total_df)

    def test_mu_recovery(self, synthetic_total_df):
        """Fitted mu should be within 0.3 of true value."""
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        assert result.params.mu == pytest.approx(TRUE_MU, abs=0.3)

    def test_alpha_positive(self, synthetic_total_df):
        """Fitted alpha should be positive."""
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        assert result.params.alpha > 0

    def test_alpha_recovery(self, synthetic_total_df):
        """Fitted alpha should be within 0.3 of true value."""
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        assert result.params.alpha == pytest.approx(TRUE_ALPHA, abs=0.3)

    def test_home_effect_sums_to_zero(self, synthetic_total_df):
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        assert result.params.home_effect.sum() == pytest.approx(0.0, abs=1e-10)

    def test_away_effect_sums_to_zero(self, synthetic_total_df):
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        assert result.params.away_effect.sum() == pytest.approx(0.0, abs=1e-10)

    def test_teams_sorted(self, synthetic_total_df):
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        assert result.params.teams == sorted(TEAMS)

    def test_predicted_mean_close_to_actual(self, synthetic_total_df):
        """Mean predicted total should be close to actual mean total."""
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        actual_mean = (
            synthetic_total_df["home_corners"] + synthetic_total_df["away_corners"]
        ).mean()
        # Predicted mean across all matches
        team_to_idx = {t: i for i, t in enumerate(result.params.teams)}
        pred_totals = []
        for _, row in synthetic_total_df.iterrows():
            hi = team_to_idx[row["home_team"]]
            ai = team_to_idx[row["away_team"]]
            mu_t = np.exp(
                result.params.mu
                + result.params.home_effect[hi]
                + result.params.away_effect[ai]
            )
            pred_totals.append(mu_t)
        pred_mean = np.mean(pred_totals)
        assert pred_mean == pytest.approx(actual_mean, rel=0.1)


class TestWarmStart:
    """Warm-start from previous fit."""

    def test_warm_start_accepted(self, synthetic_total_df):
        result1 = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        x0 = pack_total(result1.params)
        result2 = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
            x0=x0,
        )
        assert result2.converged

    def test_wrong_length_x0_ignored(self, synthetic_total_df):
        x0_bad = np.zeros(3, dtype=np.float64)
        result = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
            x0=x0_bad,
        )
        assert result.converged

    def test_warm_and_cold_agree(self, synthetic_total_df):
        cold = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
        )
        x0 = pack_total(cold.params)
        warm = fit_total_count_model(
            synthetic_total_df, "home_corners", "away_corners",
            x0=x0,
        )
        assert cold.params.mu == pytest.approx(warm.params.mu, abs=1e-3)
        assert cold.params.alpha == pytest.approx(warm.params.alpha, abs=1e-3)
