"""Integration test: 20-row real CSV → test DB, full pipeline, idempotent re-run.

Uses the sample_e0.csv fixture with mocked DB sessions.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.engine.ingest.alias_resolver import AliasResolver
from services.engine.ingest.config import IngestConfig
from services.engine.ingest.csv_loader import load_season_csv
from services.engine.ingest.csv_parser import parse_csv

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "fixtures" / "sample_e0.csv"

# Full set of teams in the sample CSV
TEAMS = {
    1: "Liverpool", 2: "Norwich City", 3: "West Ham United", 4: "Manchester City",
    5: "AFC Bournemouth", 6: "Sheffield United", 7: "Burnley", 8: "Southampton",
    9: "Crystal Palace", 10: "Everton", 11: "Leicester City",
    12: "Wolverhampton Wanderers", 13: "Watford", 14: "Brighton & Hove Albion",
    15: "Tottenham Hotspur", 16: "Aston Villa", 17: "Newcastle United",
    18: "Arsenal", 19: "Manchester United", 20: "Chelsea",
}

ALIASES = {
    ("fd_couk", "Liverpool"): 1, ("fd_couk", "Norwich"): 2,
    ("fd_couk", "West Ham"): 3, ("fd_couk", "Man City"): 4,
    ("fd_couk", "Bournemouth"): 5, ("fd_couk", "Sheffield United"): 6,
    ("fd_couk", "Burnley"): 7, ("fd_couk", "Southampton"): 8,
    ("fd_couk", "Crystal Palace"): 9, ("fd_couk", "Everton"): 10,
    ("fd_couk", "Leicester"): 11, ("fd_couk", "Wolves"): 12,
    ("fd_couk", "Watford"): 13, ("fd_couk", "Brighton"): 14,
    ("fd_couk", "Tottenham"): 15, ("fd_couk", "Aston Villa"): 16,
    ("fd_couk", "Newcastle"): 17, ("fd_couk", "Arsenal"): 18,
    ("fd_couk", "Man United"): 19, ("fd_couk", "Chelsea"): 20,
}


class TestFullPipeline:
    def test_parse_all_20_rows(self):
        """All 20 rows from the sample CSV should be parsed."""
        rows = parse_csv(SAMPLE_CSV)
        assert len(rows) == 20

    def test_all_teams_resolved(self):
        """All teams in the sample CSV should resolve via aliases."""
        resolver = AliasResolver(teams=TEAMS, aliases=ALIASES)
        rows = parse_csv(SAMPLE_CSV)

        unresolved = []
        for row in rows:
            home_id = resolver.resolve("fd_couk", row["home_team"])
            away_id = resolver.resolve("fd_couk", row["away_team"])
            if home_id is None:
                unresolved.append(row["home_team"])
            if away_id is None:
                unresolved.append(row["away_team"])

        assert len(unresolved) == 0, f"Unresolved teams: {unresolved}"

    def test_all_rows_have_goals(self):
        """All 20 rows should have valid goal counts."""
        rows = parse_csv(SAMPLE_CSV)
        for row in rows:
            assert row["ft_home_goals"] is not None
            assert row["ft_away_goals"] is not None
            assert row["ft_home_goals"] >= 0
            assert row["ft_away_goals"] >= 0

    def test_all_rows_have_stats(self):
        """All 20 rows should have complete stat columns."""
        rows = parse_csv(SAMPLE_CSV)
        stat_keys = [
            "home_shots", "away_shots", "home_corners", "away_corners",
            "home_yellows", "away_yellows", "home_reds", "away_reds",
        ]
        for row in rows:
            for key in stat_keys:
                assert row[key] is not None, f"Missing {key} in {row['home_team']} vs {row['away_team']}"

    def test_no_duplicate_fixtures(self):
        """No home/away team pair should appear twice."""
        rows = parse_csv(SAMPLE_CSV)
        pairs = set()
        for row in rows:
            pair = (row["home_team"], row["away_team"])
            assert pair not in pairs, f"Duplicate fixture: {pair}"
            pairs.add(pair)


class TestIdempotentRerun:
    @patch("services.engine.ingest.csv_loader._download_csv")
    def test_rerun_produces_same_counts(self, mock_download):
        """Running the loader twice should produce the same result."""
        mock_download.return_value = SAMPLE_CSV

        config = IngestConfig()
        config.raw_data_dir = "/tmp/test_raw"

        resolver = AliasResolver(teams=TEAMS, aliases=ALIASES)
        resolver.flush_new_aliases = MagicMock(return_value=0)

        session = MagicMock()
        session.query.return_value.filter_by.return_value.one.return_value = MagicMock(id=1)
        session.execute.return_value.rowcount = 1

        w1, s1 = load_season_csv(session, config, "2019-20", 1, 1, resolver)
        w2, s2 = load_season_csv(session, config, "2019-20", 1, 1, resolver)

        assert w1 == w2
        assert s1 == s2

    def test_resolver_caches_resolved_aliases(self):
        """Once resolved, an alias should be cached for future lookups."""
        resolver = AliasResolver(teams=TEAMS, aliases=ALIASES)

        # First resolution
        id1 = resolver.resolve("fd_couk", "Liverpool")
        # Second resolution should hit cache (same source + raw_name)
        id2 = resolver.resolve("fd_couk", "Liverpool")

        assert id1 == id2 == 1
