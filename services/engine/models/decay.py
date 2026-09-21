"""Time-decay weights for Dixon-Coles model.

Recent matches receive higher weight than older ones. The decay function is
w(t) = exp(-xi * t) where t is measured in half-weeks (days / 3.5).

Reference: Dixon & Coles (1997), Section 4.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


def time_weights(
    match_dates: NDArray[np.datetime64] | list[date],
    reference_date: date | np.datetime64,
    xi: float,
) -> NDArray[np.float64]:
    """Compute exponential decay weights for each match.

    Args:
        match_dates: date of each match
        reference_date: the most recent date (weight = 1.0)
        xi: decay rate parameter (higher = faster decay)

    Returns:
        Array of weights in [0, 1], with weight=1 at reference_date.
    """
    ref = np.datetime64(reference_date, "D")
    dates = np.asarray(match_dates, dtype="datetime64[D]")
    days_ago = (ref - dates).astype(np.float64)
    half_weeks = days_ago / 3.5
    return np.exp(-xi * half_weeks)


def half_life_days(xi: float) -> float:
    """Return the half-life in days for a given xi.

    Half-life is the number of days until weight drops to 0.5:
        exp(-xi * d/3.5) = 0.5  =>  d = 3.5 * ln(2) / xi
    """
    return 3.5 * np.log(2) / xi


def xi_from_half_life(half_life: float) -> float:
    """Return xi that produces the given half-life in days."""
    return 3.5 * np.log(2) / half_life


def optimise_xi(
    df: pd.DataFrame,
    xi_candidates: NDArray[np.float64] | None = None,
) -> tuple[float, list[tuple[float, float]]]:
    """Find the xi that minimises negative log-likelihood via grid search.

    Fits the Dixon-Coles model at each candidate xi value and returns the
    best one. This is a simple grid search — xi is a single scalar, so
    gradient-based optimisation is overkill.

    Args:
        df: DataFrame with columns: date, home_team, away_team, home_goals, away_goals
        xi_candidates: array of xi values to try; defaults to np.arange(0.001, 0.015, 0.001)

    Returns:
        (best_xi, scores) where scores is a list of (xi, nll) pairs.
    """
    import pandas as pd
    from services.engine.models.fit import fit_dixon_coles

    if xi_candidates is None:
        xi_candidates = np.arange(0.001, 0.015, 0.001)

    dates = pd.to_datetime(df["date"])
    ref_date = dates.max()
    match_dates = dates.values.astype("datetime64[D]")

    scores: list[tuple[float, float]] = []
    for xi in xi_candidates:
        weights = time_weights(match_dates, ref_date, xi=float(xi))
        result = fit_dixon_coles(df, weights=weights)
        scores.append((float(xi), result.neg_log_lik))

    best_xi, _ = min(scores, key=lambda x: x[1])
    return best_xi, scores
