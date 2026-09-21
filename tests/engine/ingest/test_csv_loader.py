"""Tests for CSV loader: idempotency, cache hit, unresolved alias skip counting."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.engine.ingest.alias_resolver import AliasResolver
from services.engine.ingest.config import IngestConfig
from services.engine.ingest.csv_loader import load_season_csv


SAMPLE_CSV = Path(__file__).resolve().parents[2] / "fixtures" / "sample_e0.csv"

# Teams present in sample_e0.csv
SAMPLE_TEAMS = {
    1: "Liverpool", 2: "Norwich City", 3: "West Ham United", 4: "Manchester City",
    5: "AFC Bournemouth", 6: "Sheffield United", 7: "Burnley", 8: "Southampton",
    9: "Crystal Palace", 10: "Everton", 11: "Leicester City",
    12: "Wolverhampton Wanderers", 13: "Watford", 14: "Brighton & Hove Albion",
    15: "Tottenham Hotspur", 16: "Aston Villa", 17: "Newcastle United",
    18: "Arsenal", 19: "Manchester United", 20: "Chelsea",
}

# Aliases matching fd_couk short names to team IDs
SAMPLE_ALIASES = {
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


@pytest.fixture
def resolver():
    r = AliasResolver(teams=SAMPLE_TEAMS, aliases=SAMPLE_ALIASES)
    r.flush_new_aliases = MagicMock(return_value=0)
    return r


@pytest.fixture
def config(tmp_path):
    cfg = IngestConfig()
    cfg.raw_data_dir = str(tmp_path / "raw")
    cfg.football_data_couk_base_url = "https://example.com"
    return cfg


class TestIdempotency:
    @patch("services.engine.ingest.csv_loader._download_csv")
    def test_double_load_same_result(self, mock_download, config, resolver):
        """Loading the same CSV twice should produce the same data."""
        mock_download.return_value = SAMPLE_CSV

        session = MagicMock()
        session.query.return_value.filter_by.return_value.one.return_value = MagicMock(id=1)
        session.execute.return_value.rowcount = 1

        w1, s1 = load_season_csv(session, config, "2019-20", 1, 1, resolver)
        w2, s2 = load_season_csv(session, config, "2019-20", 1, 1, resolver)

        # Both should process the same number of rows
        assert w1 == w2
        assert s1 == s2 == 0


class TestCacheHit:
    def test_uses_cache_when_exists(self, config, resolver, tmp_path):
        """Cached CSV should be used without re-download."""
        import shutil
        raw_dir = Path(config.raw_data_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        cache_file = raw_dir / "E0_201920.csv"
        shutil.copy(SAMPLE_CSV, cache_file)

        session = MagicMock()
        session.query.return_value.filter_by.return_value.one.return_value = MagicMock(id=1)
        session.execute.return_value.rowcount = 1

        # Should not raise — uses cached file
        w, s = load_season_csv(session, config, "2019-20", 1, 1, resolver)
        assert w == 20
        assert s == 0


class TestUnresolvedAliasSkip:
    def test_skips_unresolved_teams(self, config):
        """Rows with unresolved team names should be counted as skipped."""
        # Resolver with no aliases — everything unresolved
        resolver = AliasResolver(teams={}, aliases={})
        resolver.flush_new_aliases = MagicMock(return_value=0)

        session = MagicMock()

        with patch("services.engine.ingest.csv_loader._download_csv", return_value=SAMPLE_CSV):
            w, s = load_season_csv(session, config, "2019-20", 1, 1, resolver)

        assert w == 0
        assert s == 20  # All 20 rows skipped
