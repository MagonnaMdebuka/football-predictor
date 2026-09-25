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
    """Assign matchday (round) numbers within each season.

    A matchday is a fixture round: the set of matches where each team
    plays at most once. Matches are sorted by date within each season
    and accumulated into the current round. When a team appears for the
    second time, a new round begins.

    Requires columns: 'date', 'home_team', 'away_team', and *season_col*.

    Args:
        df: DataFrame with 'date', 'home_team', 'away_team', and season_col
        season_col: name of the season column

    Returns:
        Series of matchday numbers (1-indexed), aligned with df's index.
    """
    result = pd.Series(0, index=df.index, dtype=np.int64)

    for _season, group in df.groupby(season_col):
        sorted_group = group.sort_values("date")
        matchday = 1
        seen_teams: set[str] = set()

        for idx, row in sorted_group.iterrows():
            ht = row["home_team"]
            at = row["away_team"]
            if ht in seen_teams or at in seen_teams:
                matchday += 1
                seen_teams = set()
            seen_teams.add(ht)
            seen_teams.add(at)
            result.loc[idx] = matchday

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

    Uses team-repetition-based matchday assignment to count rounds.

    Args:
        df: DataFrame with 'date', 'home_team', 'away_team', and season_col

    Returns:
        Dict mapping season to matchday count.
    """
    matchdays = assign_matchdays(df, season_col)
    counts = {}
    for season, group in df.groupby(season_col):
        counts[str(season)] = int(matchdays.loc[group.index].max())
    return counts
