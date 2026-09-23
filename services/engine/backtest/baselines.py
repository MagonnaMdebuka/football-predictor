"""Baseline predictors for backtest comparison.

Four baselines the Dixon-Coles model is measured against:
1. Uniform — 1/3 for each outcome
2. Base rate — home/draw/away proportions from training data
3. Independent Poisson — Dixon-Coles with rho forced to 0 (no tau correction)
4. Bookmaker — Pinnacle/market average odds converted to probabilities
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from services.engine.backtest.types import BookmakerOddsCols


@dataclass(frozen=True)
class BaseRateProbs:
    """Home/draw/away proportions from a training set."""

    home: float
    draw: float
    away: float


def uniform_probabilities(n_matches: int) -> tuple[NDArray, NDArray, NDArray]:
    """Return uniform 1/3 probabilities for all matches.

    Args:
        n_matches: number of matches

    Returns:
        (p_home, p_draw, p_away) each of shape (n_matches,)
    """
    third = np.full(n_matches, 1.0 / 3.0, dtype=np.float64)
    return third.copy(), third.copy(), third.copy()


def base_rate_from_training(training_df: pd.DataFrame) -> BaseRateProbs:
    """Compute home/draw/away proportions from training data.

    Uses the 'ftr' column (Full-Time Result: H/D/A). Falls back to
    computing from home_goals/away_goals if 'ftr' is not present.

    Args:
        training_df: training data with 'ftr' or 'home_goals'/'away_goals' columns

    Returns:
        BaseRateProbs with proportions summing to 1.0.
    """
    n = len(training_df)
    if n == 0:
        return BaseRateProbs(home=1.0 / 3.0, draw=1.0 / 3.0, away=1.0 / 3.0)

    if "ftr" in training_df.columns:
        ftr = training_df["ftr"]
        n_home = (ftr == "H").sum()
        n_draw = (ftr == "D").sum()
        n_away = (ftr == "A").sum()
    else:
        hg = training_df["home_goals"]
        ag = training_df["away_goals"]
        n_home = (hg > ag).sum()
        n_draw = (hg == ag).sum()
        n_away = (hg < ag).sum()

    total = n_home + n_draw + n_away
    if total == 0:
        return BaseRateProbs(home=1.0 / 3.0, draw=1.0 / 3.0, away=1.0 / 3.0)

    return BaseRateProbs(
        home=float(n_home / total),
        draw=float(n_draw / total),
        away=float(n_away / total),
    )


def base_rate_probabilities(
    base_rate: BaseRateProbs,
    n_matches: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Broadcast base-rate proportions to arrays.

    Args:
        base_rate: training-set proportions
        n_matches: number of matches to predict

    Returns:
        (p_home, p_draw, p_away) each of shape (n_matches,)
    """
    return (
        np.full(n_matches, base_rate.home, dtype=np.float64),
        np.full(n_matches, base_rate.draw, dtype=np.float64),
        np.full(n_matches, base_rate.away, dtype=np.float64),
    )


def independent_poisson_probabilities(
    training_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    weights: NDArray[np.float64] | None = None,
    rho_bounds: tuple[float, float] = (0.0, 0.0),
    max_goals: int = 11,
) -> tuple[NDArray, NDArray, NDArray]:
    """Predict using Dixon-Coles with rho forced to zero (independent Poisson).

    This isolates the contribution of the tau correction and time decay
    by fitting the model without the low-score dependence parameter.

    Args:
        training_df: training data for fitting
        prediction_df: matches to predict (need home_team, away_team)
        weights: optional time-decay weights for training data
        rho_bounds: forced to (0, 0) to disable rho
        max_goals: grid size for score matrix

    Returns:
        (p_home, p_draw, p_away) for each prediction match
    """
    from services.engine.models.fit import fit_dixon_coles
    from services.engine.models.grid import build_grid

    result = fit_dixon_coles(training_df, weights=weights, rho_bounds=rho_bounds)

    homes = []
    draws = []
    aways = []
    for _, row in prediction_df.iterrows():
        grid = build_grid(result.params, row["home_team"], row["away_team"])
        homes.append(grid.home_win)
        draws.append(grid.draw)
        aways.append(grid.away_win)

    return (
        np.array(homes, dtype=np.float64),
        np.array(draws, dtype=np.float64),
        np.array(aways, dtype=np.float64),
    )


def bookmaker_probabilities(
    raw_jsons: list[str | None],
    cols: BookmakerOddsCols | None = None,
) -> tuple[NDArray | None, NDArray | None, NDArray | None, int]:
    """Extract bookmaker probabilities from raw JSON strings.

    Delegates to odds.batch_extract_probabilities.

    Args:
        raw_jsons: list of JSON strings from match_source_rows.raw
        cols: column name configuration

    Returns:
        (p_home, p_draw, p_away, exclusion_count)
    """
    from services.engine.backtest.odds import batch_extract_probabilities
    return batch_extract_probabilities(raw_jsons, cols)
