"""Tests for Dixon-Coles negative log-likelihood."""

import numpy as np
import pandas as pd

from services.engine.models.likelihood import neg_log_likelihood
from services.engine.models.params import DixonColesParams, pack
from tests.engine.models.conftest import (
    TRUE_ATTACK,
    TRUE_DEFENCE,
    TRUE_GAMMA,
    TRUE_MU,
    TRUE_RHO,
)


def _prepare_data(
    df: pd.DataFrame, teams: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert DataFrame to index arrays for the likelihood function."""
    team_to_idx = {t: i for i, t in enumerate(teams)}
    home_idx = df["home_team"].map(team_to_idx).values.astype(np.int_)
    away_idx = df["away_team"].map(team_to_idx).values.astype(np.int_)
    home_goals = df["home_goals"].values.astype(np.int_)
    away_goals = df["away_goals"].values.astype(np.int_)
    return home_idx, away_idx, home_goals, away_goals


class TestFiniteOutput:
    def test_returns_finite_scalar(self, synthetic_df: pd.DataFrame, true_teams: list[str]):
        home_idx, away_idx, home_goals, away_goals = _prepare_data(synthetic_df, true_teams)
        true_params = DixonColesParams(
            teams=true_teams,
            mu=TRUE_MU,
            attack=TRUE_ATTACK.copy(),
            defence=TRUE_DEFENCE.copy(),
            gamma=TRUE_GAMMA,
            rho=TRUE_RHO,
        )
        vec = pack(true_params)
        nll = neg_log_likelihood(vec, true_teams, home_idx, away_idx, home_goals, away_goals)
        assert np.isfinite(nll)

    def test_returns_positive(self, synthetic_df: pd.DataFrame, true_teams: list[str]):
        home_idx, away_idx, home_goals, away_goals = _prepare_data(synthetic_df, true_teams)
        vec = pack(DixonColesParams(
            teams=true_teams,
            mu=TRUE_MU,
            attack=TRUE_ATTACK.copy(),
            defence=TRUE_DEFENCE.copy(),
            gamma=TRUE_GAMMA,
            rho=TRUE_RHO,
        ))
        nll = neg_log_likelihood(vec, true_teams, home_idx, away_idx, home_goals, away_goals)
        assert nll > 0


class TestBetterAtTrueParams:
    def test_true_params_beat_zeros(self, synthetic_df: pd.DataFrame, true_teams: list[str]):
        """NLL at the true parameters should be lower than at all-zeros."""
        home_idx, away_idx, home_goals, away_goals = _prepare_data(synthetic_df, true_teams)

        true_vec = pack(DixonColesParams(
            teams=true_teams,
            mu=TRUE_MU,
            attack=TRUE_ATTACK.copy(),
            defence=TRUE_DEFENCE.copy(),
            gamma=TRUE_GAMMA,
            rho=TRUE_RHO,
        ))
        zero_vec = pack(DixonColesParams(
            teams=true_teams,
            mu=0.0,
            attack=np.zeros(4),
            defence=np.zeros(4),
            gamma=0.0,
            rho=0.0,
        ))

        nll_true = neg_log_likelihood(
            true_vec, true_teams, home_idx, away_idx, home_goals, away_goals
        )
        nll_zero = neg_log_likelihood(
            zero_vec, true_teams, home_idx, away_idx, home_goals, away_goals
        )
        assert nll_true < nll_zero


class TestWeights:
    def test_zero_weight_ignored(self, synthetic_df: pd.DataFrame, true_teams: list[str]):
        """Matches with weight=0 should not contribute to the likelihood."""
        home_idx, away_idx, home_goals, away_goals = _prepare_data(synthetic_df, true_teams)
        vec = pack(DixonColesParams(
            teams=true_teams,
            mu=TRUE_MU,
            attack=TRUE_ATTACK.copy(),
            defence=TRUE_DEFENCE.copy(),
            gamma=TRUE_GAMMA,
            rho=TRUE_RHO,
        ))

        # Full weights
        nll_full = neg_log_likelihood(
            vec, true_teams, home_idx, away_idx, home_goals, away_goals
        )

        # Zero out the last half of matches
        n = len(home_goals)
        weights_half = np.ones(n, dtype=np.float64)
        weights_half[n // 2 :] = 0.0
        nll_half = neg_log_likelihood(
            vec, true_teams, home_idx, away_idx, home_goals, away_goals, weights=weights_half
        )

        # Fewer matches → lower total NLL
        assert nll_half < nll_full
