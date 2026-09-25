"""Tests for matchday assignment and early-season detection."""

from __future__ import annotations

import pandas as pd

from services.engine.backtest.matchday import (
    EARLY_SEASON_MATCHDAYS,
    assign_matchdays,
    is_early_season,
    season_matchday_counts,
)


def _make_df(
    dates: list[str],
    seasons: list[str],
    home_teams: list[str] | None = None,
    away_teams: list[str] | None = None,
) -> pd.DataFrame:
    d: dict = {
        "date": pd.to_datetime(dates),
        "season": seasons,
    }
    if home_teams is not None:
        d["home_team"] = home_teams
    if away_teams is not None:
        d["away_team"] = away_teams
    return pd.DataFrame(d)


class TestAssignMatchdays:
    """Matchday numbering within seasons using team-repetition logic."""

    def test_single_season_sequential(self):
        """Matches with no team repeats stay in the same round."""
        df = _make_df(
            ["2024-08-17", "2024-08-17", "2024-08-24"],
            ["2024-25"] * 3,
            home_teams=["A", "C", "E"],
            away_teams=["B", "D", "F"],
        )
        matchdays = assign_matchdays(df)
        # All three matches involve distinct teams, so all round 1
        # even though the third is on a different date
        assert list(matchdays) == [1, 1, 1]

    def test_team_repeat_starts_new_round(self):
        """When a team appears again, a new matchday begins."""
        df = _make_df(
            ["2024-08-17", "2024-08-17", "2024-08-17"],
            ["2024-25"] * 3,
            home_teams=["A", "C", "A"],
            away_teams=["B", "D", "E"],
        )
        matchdays = assign_matchdays(df)
        # First two: A-B, C-D (no overlap) => matchday 1
        # Third: A-E (A already seen) => matchday 2
        assert list(matchdays) == [1, 1, 2]

    def test_opening_weekend_is_one_matchday(self):
        """A full PL-style round of 10 matches across Fri-Mon is one matchday."""
        teams = [
            ("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"), ("I", "J"),
            ("K", "L"), ("M", "N"), ("O", "P"), ("Q", "R"), ("S", "T"),
        ]
        # Spread across 3 days: Fri (3), Sat (5), Sun (2)
        dates = (
            ["2024-08-16"] * 3 + ["2024-08-17"] * 5 + ["2024-08-18"] * 2
        )
        df = _make_df(
            dates,
            ["2024-25"] * 10,
            home_teams=[t[0] for t in teams],
            away_teams=[t[1] for t in teams],
        )
        matchdays = assign_matchdays(df)
        assert list(matchdays) == [1] * 10

    def test_60_matches_in_first_6_matchdays(self):
        """6 rounds of 10 PL-style matches = 60 matches in first 6 matchdays."""
        all_teams = [chr(65 + i) for i in range(20)]  # A-T
        rows = []
        base_date = pd.Timestamp("2024-08-17")
        for round_num in range(6):
            for i in range(0, 20, 2):
                rows.append({
                    "date": base_date + pd.Timedelta(days=round_num * 7),
                    "season": "2024-25",
                    "home_team": all_teams[i],
                    "away_team": all_teams[i + 1],
                })
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        matchdays = assign_matchdays(df)
        early = matchdays[matchdays <= 6]
        assert len(early) == 60

    def test_seasons_numbered_independently(self):
        """Each season starts at matchday 1."""
        df = _make_df(
            ["2024-08-17", "2024-08-24", "2025-08-16", "2025-08-23"],
            ["2024-25", "2024-25", "2025-26", "2025-26"],
            home_teams=["A", "A", "A", "A"],
            away_teams=["B", "C", "B", "C"],
        )
        matchdays = assign_matchdays(df)
        assert list(matchdays) == [1, 2, 1, 2]

    def test_handles_unsorted_dates(self):
        """Matchdays assigned by date order, not DataFrame order."""
        df = _make_df(
            ["2024-08-24", "2024-08-17"],
            ["2024-25", "2024-25"],
            home_teams=["A", "A"],
            away_teams=["B", "C"],
        )
        matchdays = assign_matchdays(df)
        # 17th is earlier, so matchday 1 assigned to row index 1
        # 24th gets matchday 2 (A appears again)
        assert matchdays.iloc[0] == 2  # 24th
        assert matchdays.iloc[1] == 1  # 17th

    def test_preserves_index_alignment(self):
        df = _make_df(
            ["2024-08-17", "2024-08-24"],
            ["2024-25", "2024-25"],
            home_teams=["A", "A"],
            away_teams=["B", "C"],
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
            home_teams=["A", "C", "A"],
            away_teams=["B", "D", "E"],
        )
        counts = season_matchday_counts(df)
        # A-B, C-D are round 1; A-E triggers round 2
        assert counts == {"2024-25": 2}

    def test_multiple_seasons(self):
        df = _make_df(
            ["2024-08-17", "2024-08-24", "2025-08-16"],
            ["2024-25", "2024-25", "2025-26"],
            home_teams=["A", "A", "A"],
            away_teams=["B", "C", "B"],
        )
        counts = season_matchday_counts(df)
        assert counts == {"2024-25": 2, "2025-26": 1}
