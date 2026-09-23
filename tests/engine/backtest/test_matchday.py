"""Tests for matchday assignment and early-season detection."""

from __future__ import annotations

import pandas as pd

from services.engine.backtest.matchday import (
    EARLY_SEASON_MATCHDAYS,
    assign_matchdays,
    is_early_season,
    season_matchday_counts,
)


def _make_df(dates: list[str], seasons: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "season": seasons,
    })


class TestAssignMatchdays:
    """Matchday numbering within seasons."""

    def test_single_season_sequential(self):
        df = _make_df(
            ["2024-08-17", "2024-08-24", "2024-08-31"],
            ["2024-25"] * 3,
        )
        matchdays = assign_matchdays(df)
        assert list(matchdays) == [1, 2, 3]

    def test_same_date_same_matchday(self):
        """Multiple matches on the same date share a matchday."""
        df = _make_df(
            ["2024-08-17", "2024-08-17", "2024-08-24"],
            ["2024-25"] * 3,
        )
        matchdays = assign_matchdays(df)
        assert matchdays.iloc[0] == matchdays.iloc[1]
        assert matchdays.iloc[2] == 2

    def test_seasons_numbered_independently(self):
        """Each season starts at matchday 1."""
        df = _make_df(
            ["2024-08-17", "2024-08-24", "2025-08-16", "2025-08-23"],
            ["2024-25", "2024-25", "2025-26", "2025-26"],
        )
        matchdays = assign_matchdays(df)
        assert list(matchdays) == [1, 2, 1, 2]

    def test_handles_unsorted_dates(self):
        """Matchdays assigned by date order, not DataFrame order."""
        df = _make_df(
            ["2024-08-24", "2024-08-17"],
            ["2024-25", "2024-25"],
        )
        matchdays = assign_matchdays(df)
        assert matchdays.iloc[0] == 2  # 24th is the 2nd matchday
        assert matchdays.iloc[1] == 1  # 17th is the 1st

    def test_preserves_index_alignment(self):
        df = _make_df(
            ["2024-08-17", "2024-08-24"],
            ["2024-25", "2024-25"],
        )
        df.index = [10, 20]
        matchdays = assign_matchdays(df)
        assert matchdays.loc[10] == 1
        assert matchdays.loc[20] == 2


class TestIsEarlySeason:
    """Early-season flag based on matchday threshold."""

    def test_default_threshold_is_six(self):
        assert EARLY_SEASON_MATCHDAYS == 6

    def test_within_threshold(self):
        matchdays = pd.Series([1, 3, 6, 7, 10])
        early = is_early_season(matchdays)
        assert list(early) == [True, True, True, False, False]

    def test_custom_threshold(self):
        matchdays = pd.Series([1, 2, 3, 4])
        early = is_early_season(matchdays, threshold=2)
        assert list(early) == [True, True, False, False]


class TestSeasonMatchdayCounts:
    """Count distinct matchdays per season."""

    def test_single_season(self):
        df = _make_df(
            ["2024-08-17", "2024-08-17", "2024-08-24"],
            ["2024-25"] * 3,
        )
        counts = season_matchday_counts(df)
        assert counts == {"2024-25": 2}

    def test_multiple_seasons(self):
        df = _make_df(
            ["2024-08-17", "2024-08-24", "2025-08-16"],
            ["2024-25", "2024-25", "2025-26"],
        )
        counts = season_matchday_counts(df)
        assert counts == {"2024-25": 2, "2025-26": 1}
