"""Fit the Dixon-Coles model via L-BFGS-B optimisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from services.engine.models.likelihood import neg_log_likelihood
from services.engine.models.params import DixonColesParams, unpack, vector_length


@dataclass(frozen=True)
class FitResult:
    """Outcome of a Dixon-Coles model fit.

    Attributes:
        params: fitted model parameters
        neg_log_lik: final negative log-likelihood value
        converged: whether the optimiser reported convergence
        n_matches: number of matches used in fitting
    """

    params: DixonColesParams
    neg_log_lik: float
    converged: bool
    n_matches: int


def fit_dixon_coles(
    df: pd.DataFrame,
    weights: NDArray[np.float64] | None = None,
    rho_bounds: tuple[float, float] = (-0.5, 0.5),
    max_iter: int = 200,
) -> FitResult:
    """Fit the Dixon-Coles model to match data.

    Args:
        df: DataFrame with columns: home_team, away_team, home_goals, away_goals
        weights: optional per-match weights (e.g. from time decay)
        rho_bounds: box bounds for the rho parameter
        max_iter: maximum L-BFGS-B iterations

    Returns:
        FitResult with fitted parameters and diagnostics.
    """
    # Build team list (sorted for deterministic ordering)
    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    n = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    # Convert to index arrays
    home_idx = df["home_team"].map(team_to_idx).values.astype(np.int_)
    away_idx = df["away_team"].map(team_to_idx).values.astype(np.int_)
    home_goals = df["home_goals"].values.astype(np.int_)
    away_goals = df["away_goals"].values.astype(np.int_)

    # Initial parameter vector
    vec_len = vector_length(n)
    x0 = np.zeros(vec_len, dtype=np.float64)
    x0[0] = 0.25       # mu (home advantage)
    x0[2 * n - 1] = 0.20  # gamma (scoring rate)
    x0[2 * n] = -0.10     # rho (dependence)

    # Bounds: only rho is bounded; everything else is free
    bounds = [(None, None)] * vec_len
    bounds[2 * n] = rho_bounds

    result = minimize(
        neg_log_likelihood,
        x0,
        args=(teams, home_idx, away_idx, home_goals, away_goals, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )

    params = unpack(result.x, teams)

    return FitResult(
        params=params,
        neg_log_lik=float(result.fun),
        converged=result.success,
        n_matches=len(home_goals),
    )
