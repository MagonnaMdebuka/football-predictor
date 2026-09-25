"""Poisson goal model: expectancy calculation and probability mass functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson as poisson_dist


def goal_expectancy(
    attack_h: float | NDArray[np.float64],
    defence_a: float | NDArray[np.float64],
    attack_a: float | NDArray[np.float64],
    defence_h: float | NDArray[np.float64],
    mu: float,
    gamma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected goals for home and away teams.

    lambda_home = exp(mu + gamma + attack_h + defence_a)
    lambda_away = exp(mu + attack_a + defence_h)

    mu is the league scoring-rate intercept (appears in both);
    gamma is the home-advantage parameter (home only).

    Returns:
        (lambda_home, lambda_away) as numpy arrays.
    """
    lambda_home = np.exp(mu + gamma + attack_h + defence_a)
    lambda_away = np.exp(mu + attack_a + defence_h)
    return np.atleast_1d(lambda_home), np.atleast_1d(lambda_away)


def poisson_pmf(k: int | NDArray[np.int_], lam: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Poisson probability mass function P(X=k | lambda=lam)."""
    return poisson_dist.pmf(k, lam)


def poisson_log_pmf(
    k: int | NDArray[np.int_], lam: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Log of Poisson PMF: log P(X=k | lambda=lam)."""
    return poisson_dist.logpmf(k, lam)
