"""Matchday assignment and early-season detection.

Assigns sequential matchday numbers within each season based on distinct
match dates, and flags the first N matchdays as 'early season' for
separate metric reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARLY_SEASON_MATCHDAYS = 6


def assign_matchdays(df: pd.DataFrame, season_col: str = "season") -> pd.Series:
    """Assign matchday numbers within each season.

    Matchdays are numbered sequentially based on distinct dates within
    each season. All matches on the same date in the same season share
    the same matchday number.

    Args:
        df: DataFrame with 'date' and season_col columns
        season_col: name of the season column

    Returns:
        Series of matchday numbers (1-indexed), aligned with df's index.
    """
    result = pd.Series(0, index=df.index, dtype=np.int64)

    for season, group in df.groupby(season_col):
        dates = group["date"].dt.normalize()
        unique_dates = sorted(dates.unique())
        date_to_matchday = {d: i + 1 for i, d in enumerate(unique_dates)}
        matchdays = dates.map(date_to_matchday)
        result.loc[group.index] = matchdays.values

    return result


def is_early_season(matchdays: pd.Series, threshold: int = EARLY_SEASON_MATCHDAYS) -> pd.Series:
    """Return a boolean Series indicating early-season matches.

    Args:
        matchdays: matchday numbers (1-indexed)
        threshold: matchdays at or below this are considered early season

    Returns:
        Boolean Series (True = early season).
    """
    return matchdays <= threshold


def season_matchday_counts(df: pd.DataFrame, season_col: str = "season") -> dict[str, int]:
    """Count the number of distinct matchdays per season.

    Args:
        df: DataFrame with 'date' and season_col columns

    Returns:
        Dict mapping season to matchday count.
    """
    counts = {}
    for season, group in df.groupby(season_col):
        unique_dates = group["date"].dt.normalize().nunique()
        counts[str(season)] = int(unique_dates)
    return counts
