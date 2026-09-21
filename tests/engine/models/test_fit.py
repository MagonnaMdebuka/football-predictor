"""Tests for Dixon-Coles model fitting."""

import numpy as np
import pandas as pd
import pytest

from services.engine.models.fit import fit_dixon_coles
from tests.engine.models.conftest import (
    TRUE_ATTACK,
    TRUE_GAMMA,
    TRUE_MU,
    TRUE_RHO,
)


class TestSyntheticParameterRecovery:
    """Fit on synthetic data and check recovered params are close to true values."""

    def test_converges(self, synthetic_df: pd.DataFrame):
        result = fit_dixon_coles(synthetic_df)
        assert result.converged

    def test_mu_recovery(self, synthetic_df: pd.DataFrame):
        result = fit_dixon_coles(synthetic_df)
        assert result.params.mu == pytest.approx(TRUE_MU, abs=0.15)

    def test_gamma_recovery(self, synthetic_df: pd.DataFrame):
        result = fit_dixon_coles(synthetic_df)
        assert result.params.gamma == pytest.approx(TRUE_GAMMA, abs=0.15)

    def test_rho_recovery(self, synthetic_df: pd.DataFrame):
        result = fit_dixon_coles(synthetic_df)
        assert result.params.rho == pytest.approx(TRUE_RHO, abs=0.15)

    def test_attack_ranking_preserved(self, synthetic_df: pd.DataFrame):
        """The strongest attacker should still rank highest after fitting."""
        result = fit_dixon_coles(synthetic_df)
        true_order = np.argsort(TRUE_ATTACK)[::-1]
        fitted_order = np.argsort(result.params.attack)[::-1]
        # Top attacker should be correctly identified
        assert fitted_order[0] == true_order[0]


class TestSumToZeroConstraint:
    def test_attack_sums_to_zero(self, synthetic_df: pd.DataFrame):
        result = fit_dixon_coles(synthetic_df)
        assert result.params.attack.sum() == pytest.approx(0.0, abs=1e-10)

    def test_defence_sums_to_zero(self, synthetic_df: pd.DataFrame):
        result = fit_dixon_coles(synthetic_df)
        assert result.params.defence.sum() == pytest.approx(0.0, abs=1e-10)


class TestSmokeTestSampleCSV:
    """Fit on the 20-row sample CSV — just check it converges and returns sane values."""

    def test_converges_on_real_data(self, sample_csv_df: pd.DataFrame):
        result = fit_dixon_coles(sample_csv_df)
        assert result.converged

    def test_finite_nll(self, sample_csv_df: pd.DataFrame):
        result = fit_dixon_coles(sample_csv_df)
        assert np.isfinite(result.neg_log_lik)

    def test_correct_team_count(self, sample_csv_df: pd.DataFrame):
        result = fit_dixon_coles(sample_csv_df)
        n_teams = sample_csv_df["home_team"].nunique()
        assert len(result.params.teams) == n_teams

    def test_rho_within_bounds(self, sample_csv_df: pd.DataFrame):
        result = fit_dixon_coles(sample_csv_df)
        assert -0.5 <= result.params.rho <= 0.5

    def test_n_matches_correct(self, sample_csv_df: pd.DataFrame):
        result = fit_dixon_coles(sample_csv_df)
        assert result.n_matches == len(sample_csv_df)
