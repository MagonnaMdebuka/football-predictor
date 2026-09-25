"""Fit the total count NB2 model via L-BFGS-B optimisation.

Fits total counts directly (home + away) as a single NB2 distribution
with per-team home/away effects. Used for match-level O/U markets where
the independence assumption of convolution is violated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from services.engine.models.total_count_likelihood import neg_log_likelihood_total_count
from services.engine.models.total_count_params import (
    TotalCountModelParams,
    total_vector_length,
    unpack_total,
)


@dataclass(frozen=True)
class TotalCountFitResult:
    """Outcome of a total count model fit.

    Attributes:
        params: fitted model parameters
        neg_log_lik: final negative log-likelihood value
        converged: whether the optimiser reported convergence
        n_matches: number of matches used in fitting
    """

    params: TotalCountModelParams
    neg_log_lik: float
    converged: bool
    n_matches: int


def fit_total_count_model(
    df: pd.DataFrame,
    target_home_col: str,
    target_away_col: str,
    weights: NDArray[np.float64] | None = None,
    alpha_bounds: tuple[float, float] = (1e-6, 5.0),
    max_iter: int = 200,
    x0: NDArray[np.float64] | None = None,
) -> TotalCountFitResult:
    """Fit a total count NB2 model to match data.

    Args:
        df: DataFrame with columns: home_team, away_team, and the target columns.
        target_home_col: column name for home team's count (e.g. 'home_corners')
        target_away_col: column name for away team's count (e.g. 'away_corners')
        weights: optional per-match weights (e.g. from time decay)
        alpha_bounds: box bounds for the NB2 alpha parameter
        max_iter: maximum L-BFGS-B iterations
        x0: optional initial parameter vector for warm-starting

    Returns:
        TotalCountFitResult with fitted parameters and diagnostics.
    """
    # Build team list (sorted for deterministic ordering)
    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    n = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    # Convert to index arrays
    home_idx = df["home_team"].map(team_to_idx).values.astype(np.int_)
    away_idx = df["away_team"].map(team_to_idx).values.astype(np.int_)
    total_counts = (
        df[target_home_col].values.astype(np.int_)
        + df[target_away_col].values.astype(np.int_)
    )

    # Initial parameter vector
    vec_len = total_vector_length(n)
    if x0 is not None and len(x0) == vec_len:
        x0_vec = x0.copy()
    else:
        x0_vec = np.zeros(vec_len, dtype=np.float64)
        # mu: log of the mean total count
        mean_total = max(float(total_counts.mean()), 0.01)
        x0_vec[2 * (n - 1)] = np.log(mean_total)
        x0_vec[2 * (n - 1) + 1] = 0.1  # alpha (slight overdispersion)

    # Bounds: only alpha is bounded; everything else is free
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * vec_len
    bounds[2 * (n - 1) + 1] = alpha_bounds

    result = minimize(
        neg_log_likelihood_total_count,
        x0_vec,
        args=(teams, home_idx, away_idx, total_counts, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )

    params = unpack_total(result.x, teams)

    return TotalCountFitResult(
        params=params,
        neg_log_lik=float(result.fun),
        converged=result.success,
        n_matches=len(total_counts),
    )
