"""Vectorised negative log-likelihood for Dixon-Coles model.

This module provides the objective function that scipy.optimize.minimize
will call. It is designed for speed: numpy vectorised operations, no
per-match Python loops.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from services.engine.models.params import unpack
from services.engine.models.poisson import goal_expectancy, poisson_log_pmf
from services.engine.models.tau import tau

# Minimum lambda to avoid log(0)
_LAMBDA_FLOOR = 1e-10


def neg_log_likelihood(
    vec: NDArray[np.float64],
    teams: list[str],
    home_idx: NDArray[np.int_],
    away_idx: NDArray[np.int_],
    home_goals: NDArray[np.int_],
    away_goals: NDArray[np.int_],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute the weighted negative log-likelihood.

    Args:
        vec: packed parameter vector (see params.pack)
        teams: ordered team list
        home_idx: index into teams for each match's home team
        away_idx: index into teams for each match's away team
        home_goals: goals scored by home team per match
        away_goals: goals scored by away team per match
        weights: per-match weights (e.g. from time decay); default all 1.0

    Returns:
        Scalar negative log-likelihood (lower is better fit).
    """
    params = unpack(vec, teams)
    n_matches = len(home_goals)

    if weights is None:
        weights = np.ones(n_matches, dtype=np.float64)

    # Look up team parameters for each match
    attack_h = params.attack[home_idx]
    defence_h = params.defence[home_idx]
    attack_a = params.attack[away_idx]
    defence_a = params.defence[away_idx]

    # Expected goals
    lam_h, lam_a = goal_expectancy(
        attack_h, defence_a, attack_a, defence_h,
        mu=params.mu, gamma=params.gamma,
    )

    # Floor lambdas to avoid log(0)
    lam_h = np.maximum(lam_h, _LAMBDA_FLOOR)
    lam_a = np.maximum(lam_a, _LAMBDA_FLOOR)

    # Independent Poisson log-likelihood
    log_lik = poisson_log_pmf(home_goals, lam_h) + poisson_log_pmf(away_goals, lam_a)

    # Dixon-Coles tau correction (in log space)
    tau_vals = tau(home_goals, away_goals, lam_h, lam_a, params.rho)
    tau_vals = np.maximum(tau_vals, _LAMBDA_FLOOR)
    log_lik += np.log(tau_vals)

    # Weighted sum → negate for minimisation
    return -float(np.sum(weights * log_lik))
