"""Vectorised negative log-likelihood for NB2 count model.

Objective function for scipy.optimize.minimize. Supports an optional
referee effect for the cards model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from services.engine.models.count_params import unpack
from services.engine.models.negbin import count_expectancy, negbin_log_pmf

# Minimum mu to avoid log(0)
_MU_FLOOR = 1e-10


def neg_log_likelihood_count(
    vec: NDArray[np.float64],
    teams: list[str],
    home_idx: NDArray[np.int_],
    away_idx: NDArray[np.int_],
    home_counts: NDArray[np.int_],
    away_counts: NDArray[np.int_],
    weights: NDArray[np.float64] | None = None,
    referees: list[str] | None = None,
    ref_idx: NDArray[np.int_] | None = None,
) -> float:
    """Compute the weighted negative log-likelihood for a count model.

    Args:
        vec: packed parameter vector (see count_params.pack)
        teams: ordered team list
        home_idx: index into teams for each match's home team
        away_idx: index into teams for each match's away team
        home_counts: count for home team per match (corners, booking points)
        away_counts: count for away team per match
        weights: per-match weights (e.g. from time decay); default all 1.0
        referees: ordered referee list (for cards model); None for corners
        ref_idx: index into referees for each match's referee; None for corners

    Returns:
        Scalar negative log-likelihood (lower is better fit).
    """
    params = unpack(vec, teams, referees)
    n_matches = len(home_counts)

    if weights is None:
        weights = np.ones(n_matches, dtype=np.float64)

    # Look up team parameters for each match
    attack_h = params.attack[home_idx]
    defence_h = params.defence[home_idx]
    attack_a = params.attack[away_idx]
    defence_a = params.defence[away_idx]

    # Referee effects (same referee for both sides in a match)
    ref_eff = np.zeros(n_matches, dtype=np.float64)
    if referees and ref_idx is not None and len(params.referee_effect) > 0:
        ref_eff = params.referee_effect[ref_idx]

    # Expected counts
    mu_h, mu_a = count_expectancy(
        attack_h, defence_a, attack_a, defence_h,
        mu=params.mu, gamma=params.gamma,
        ref_effect_h=ref_eff, ref_effect_a=ref_eff,
    )

    # Floor mus to avoid log(0)
    mu_h = np.maximum(mu_h, _MU_FLOOR)
    mu_a = np.maximum(mu_a, _MU_FLOOR)

    # NB2 log-likelihood (both home and away counts)
    log_lik = (
        negbin_log_pmf(home_counts, mu_h, params.alpha)
        + negbin_log_pmf(away_counts, mu_a, params.alpha)
    )

    # Weighted sum, negated for minimisation
    return -float(np.sum(weights * log_lik))
