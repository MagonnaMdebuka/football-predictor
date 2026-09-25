"""Negative binomial count model: expectancy and probability functions.

NB2 parameterisation: variance = mu + alpha * mu^2.
When alpha < 1e-10, falls back to Poisson to avoid numerical issues.

scipy.stats.nbinom uses (n, p):
    n = 1/alpha
    p = n / (n + mu) = 1 / (1 + alpha * mu)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import nbinom as nbinom_dist
from scipy.stats import poisson as poisson_dist

# Threshold below which alpha is treated as zero (Poisson fallback)
_ALPHA_FLOOR = 1e-10

# Minimum mu to avoid log(0) or division by zero
_MU_FLOOR = 1e-10


def count_expectancy(
    attack_h: float | NDArray[np.float64],
    defence_a: float | NDArray[np.float64],
    attack_a: float | NDArray[np.float64],
    defence_h: float | NDArray[np.float64],
    mu: float,
    gamma: float,
    ref_effect_h: float | NDArray[np.float64] = 0.0,
    ref_effect_a: float | NDArray[np.float64] = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected counts for home and away teams.

    mu_home = exp(mu + gamma + attack_h + defence_a + ref_effect_h)
    mu_away = exp(mu + attack_a + defence_h + ref_effect_a)

    ref_effect_h and ref_effect_a are the referee effect for the home and
    away side respectively. For corners, these are always 0. For cards,
    both sides get the same referee offset (same referee officiates both teams).

    Returns:
        (mu_home, mu_away) as numpy arrays.
    """
    mu_home = np.exp(mu + gamma + attack_h + defence_a + ref_effect_h)
    mu_away = np.exp(mu + attack_a + defence_h + ref_effect_a)
    return np.atleast_1d(mu_home), np.atleast_1d(mu_away)


def _nb_params(
    mu_val: float | NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert (mu, alpha) to scipy nbinom (n, p) parameters.

    n = 1/alpha
    p = n / (n + mu)
    """
    mu_val = np.maximum(np.asarray(mu_val, dtype=np.float64), _MU_FLOOR)
    n = 1.0 / alpha
    p = n / (n + mu_val)
    return np.atleast_1d(np.broadcast_to(n, mu_val.shape).copy()), np.atleast_1d(p)


def negbin_pmf(
    k: int | NDArray[np.int_],
    mu_val: float | NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """NB2 probability mass function P(X=k | mu, alpha).

    Falls back to Poisson when alpha < 1e-10.
    """
    if alpha < _ALPHA_FLOOR:
        mu_val = np.maximum(np.asarray(mu_val, dtype=np.float64), _MU_FLOOR)
        return np.atleast_1d(poisson_dist.pmf(k, mu_val))
    n, p = _nb_params(mu_val, alpha)
    return np.atleast_1d(nbinom_dist.pmf(k, n, p))


def negbin_log_pmf(
    k: int | NDArray[np.int_],
    mu_val: float | NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Log of NB2 PMF: log P(X=k | mu, alpha).

    Falls back to Poisson when alpha < 1e-10.
    """
    if alpha < _ALPHA_FLOOR:
        mu_val = np.maximum(np.asarray(mu_val, dtype=np.float64), _MU_FLOOR)
        return np.atleast_1d(poisson_dist.logpmf(k, mu_val))
    n, p = _nb_params(mu_val, alpha)
    return np.atleast_1d(nbinom_dist.logpmf(k, n, p))


def negbin_cdf(
    k: int | NDArray[np.int_],
    mu_val: float | NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """NB2 cumulative distribution function P(X <= k | mu, alpha).

    Falls back to Poisson when alpha < 1e-10.
    """
    if alpha < _ALPHA_FLOOR:
        mu_val = np.maximum(np.asarray(mu_val, dtype=np.float64), _MU_FLOOR)
        return np.atleast_1d(poisson_dist.cdf(k, mu_val))
    n, p = _nb_params(mu_val, alpha)
    return np.atleast_1d(nbinom_dist.cdf(k, n, p))
