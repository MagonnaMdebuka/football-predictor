"""Tests for CSV parser: BOM, encoding, blank rows, date formats, column mapping."""

import tempfile
from datetime import date
from pathlib import Path

import pytest

from services.engine.ingest.csv_parser import compute_checksum, parse_csv


SAMPLE_CSV = Path(__file__).resolve().parents[2] / "fixtures" / "sample_e0.csv"


class TestParseCSV:
    def test_parse_sample_csv(self):
        rows = parse_csv(SAMPLE_CSV)
        assert len(rows) == 20

    def test_first_row_has_correct_teams(self):
        rows = parse_csv(SAMPLE_CSV)
        assert rows[0]["home_team"] == "Liverpool"
        assert rows[0]["away_team"] == "Norwich"

    def test_first_row_has_correct_goals(self):
        rows = parse_csv(SAMPLE_CSV)
        assert rows[0]["ft_home_goals"] == 4
        assert rows[0]["ft_away_goals"] == 1
        assert rows[0]["ht_home_goals"] == 4
        assert rows[0]["ht_away_goals"] == 0

    def test_first_row_has_stats(self):
        rows = parse_csv(SAMPLE_CSV)
        assert rows[0]["home_shots"] == 15
        assert rows[0]["away_shots"] == 7
        assert rows[0]["home_corners"] == 9
        assert rows[0]["away_corners"] == 2

    def test_date_parsing_dd_mm_yyyy(self):
        rows = parse_csv(SAMPLE_CSV)
        assert rows[0]["date"] == date(2019, 8, 9)

    def test_kickoff_utc_with_time(self):
        rows = parse_csv(SAMPLE_CSV)
        # 09/08/2019 20:00 BST = 19:00 UTC (August is BST)
        assert rows[0]["kickoff_utc"].hour == 19
        assert rows[0]["kickoff_time_known"] is True

    def test_referee_parsed(self):
        rows = parse_csv(SAMPLE_CSV)
        assert rows[0]["referee"] == "M Oliver"

    def test_raw_dict_included(self):
        rows = parse_csv(SAMPLE_CSV)
        assert "raw" in rows[0]
        assert rows[0]["raw"]["HomeTeam"] == "Liverpool"

    def test_bom_handling(self, tmp_path):
        """UTF-8 BOM should be handled transparently."""
        csv_content = (
            "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "01/01/2024,15:00,Arsenal,Chelsea,2,1\n"
        )
        csv_file = tmp_path / "bom_test.csv"
        csv_file.write_bytes(b"\xef\xbb\xbf" + csv_content.encode("utf-8"))

        rows = parse_csv(csv_file)
        assert len(rows) == 1
        assert rows[0]["home_team"] == "Arsenal"

    def test_latin1_encoding(self, tmp_path):
        """Latin-1 encoded files should be parsed correctly."""
        csv_content = "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n01/01/2024,15:00,Atl\xe9tico,Valencia,1,0\n"
        csv_file = tmp_path / "latin1_test.csv"
        csv_file.write_bytes(csv_content.encode("latin-1"))

        rows = parse_csv(csv_file)
        assert len(rows) == 1

    def test_blank_rows_skipped(self, tmp_path):
        """Blank rows should be skipped without error."""
        csv_content = (
            "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "01/01/2024,15:00,Arsenal,Chelsea,2,1\n"
            ",,,,,,\n"
            "02/01/2024,15:00,Liverpool,Everton,3,0\n"
        )
        csv_file = tmp_path / "blank_test.csv"
        csv_file.write_text(csv_content)

        rows = parse_csv(csv_file)
        assert len(rows) == 2

    def test_dd_mm_yy_date_format(self, tmp_path):
        """Two-digit year format should be parsed correctly."""
        csv_content = (
            "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "01/01/24,15:00,Arsenal,Chelsea,2,1\n"
        )
        csv_file = tmp_path / "yy_date.csv"
        csv_file.write_text(csv_content)

        rows = parse_csv(csv_file)
        assert rows[0]["date"] == date(2024, 1, 1)

    def test_missing_time_column(self, tmp_path):
        """Missing Time value should default to 15:00 with kickoff_time_known=False."""
        csv_content = (
            "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "01/01/2024,,Arsenal,Chelsea,2,1\n"
        )
        csv_file = tmp_path / "no_time.csv"
        csv_file.write_text(csv_content)

        rows = parse_csv(csv_file)
        assert rows[0]["kickoff_time_known"] is False

    def test_column_mapping(self):
        """All expected stat columns should be present in parsed output."""
        rows = parse_csv(SAMPLE_CSV)
        row = rows[0]
        expected_keys = [
            "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
            "home_fouls", "away_fouls", "home_corners", "away_corners",
            "home_yellows", "away_yellows", "home_reds", "away_reds",
        ]
        for key in expected_keys:
            assert key in row, f"Missing key: {key}"


class TestChecksum:
    def test_checksum_deterministic(self):
        c1 = compute_checksum(SAMPLE_CSV)
        c2 = compute_checksum(SAMPLE_CSV)
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex
