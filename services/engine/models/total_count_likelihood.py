"""Vectorised negative log-likelihood for the total count NB2 model.

Objective function for scipy.optimize.minimize. Fits total counts directly
(no convolution) with per-team home/away effects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from services.engine.models.negbin import negbin_log_pmf
from services.engine.models.total_count_params import unpack_total

# Minimum mu to avoid log(0)
_MU_FLOOR = 1e-10


def neg_log_likelihood_total_count(
    vec: NDArray[np.float64],
    teams: list[str],
    home_idx: NDArray[np.int_],
    away_idx: NDArray[np.int_],
    total_counts: NDArray[np.int_],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute the weighted negative log-likelihood for a total count model.

    mu_total = exp(mu + home_effect[h] + away_effect[a])

    Args:
        vec: packed parameter vector (see total_count_params.pack_total)
        teams: ordered team list
        home_idx: index into teams for each match's home team
        away_idx: index into teams for each match's away team
        total_counts: total count per match (e.g. home_corners + away_corners)
        weights: per-match weights (e.g. from time decay); default all 1.0

    Returns:
        Scalar negative log-likelihood (lower is better fit).
    """
    params = unpack_total(vec, teams)
    n_matches = len(total_counts)

    if weights is None:
        weights = np.ones(n_matches, dtype=np.float64)

    # Expected total count per match
    mu_total = np.exp(
        params.mu + params.home_effect[home_idx] + params.away_effect[away_idx]
    )
    mu_total = np.maximum(mu_total, _MU_FLOOR)

    # Single NB2 log-likelihood per match (on totals)
    log_lik = negbin_log_pmf(total_counts, mu_total, params.alpha)

    # Weighted sum, negated for minimisation
    return -float(np.sum(weights * log_lik))
