"""Tests for normalisation: timezone, accent stripping, FC removal, season codes, no_crowd."""

from datetime import date, timezone

import pytest

from services.engine.ingest.normalise import (
    is_no_crowd,
    normalise_team_name,
    parse_csv_date,
    parse_kickoff_utc,
    season_label_from_date,
)


class TestNormaliseTeamName:
    def test_lowercase(self):
        assert normalise_team_name("ARSENAL") == "arsenal"

    def test_strip_fc(self):
        assert normalise_team_name("Arsenal FC") == "arsenal"

    def test_strip_afc(self):
        assert normalise_team_name("AFC Bournemouth") == "bournemouth"

    def test_accent_stripping(self):
        result = normalise_team_name("Atlético Madrid")
        assert result == "atletico madrid"

    def test_whitespace_collapse(self):
        assert normalise_team_name("  Manchester   United  ") == "manchester united"

    def test_preserves_content(self):
        assert normalise_team_name("Wolverhampton Wanderers") == "wolverhampton wanderers"


class TestParseKickoffUTC:
    def test_bst_to_utc(self):
        """20:00 BST (August) should become 19:00 UTC."""
        dt, known = parse_kickoff_utc(date(2024, 8, 17), "20:00")
        assert dt.tzinfo == timezone.utc
        assert dt.hour == 19
        assert known is True

    def test_gmt_to_utc(self):
        """15:00 GMT (January) should stay 15:00 UTC."""
        dt, known = parse_kickoff_utc(date(2024, 1, 13), "15:00")
        assert dt.hour == 15
        assert known is True

    def test_missing_time(self):
        """No time should default to 15:00 UK time."""
        dt, known = parse_kickoff_utc(date(2024, 1, 13), None)
        assert dt.hour == 15  # GMT in January
        assert known is False

    def test_empty_time_string(self):
        dt, known = parse_kickoff_utc(date(2024, 1, 13), "")
        assert known is False

    def test_bst_boundary(self):
        """Late October — clocks go back. Match at 15:00 on BST date."""
        dt, known = parse_kickoff_utc(date(2024, 10, 5), "15:00")
        assert dt.hour == 14  # BST = UTC+1


class TestParseCSVDate:
    def test_dd_mm_yyyy(self):
        assert parse_csv_date("17/08/2024") == date(2024, 8, 17)

    def test_dd_mm_yy(self):
        assert parse_csv_date("17/08/24") == date(2024, 8, 17)

    def test_leading_zero(self):
        assert parse_csv_date("01/01/2024") == date(2024, 1, 1)

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            parse_csv_date("not-a-date")


class TestSeasonLabelFromDate:
    def test_august_is_new_season(self):
        assert season_label_from_date(date(2024, 8, 17)) == "2024-25"

    def test_january_is_same_season(self):
        assert season_label_from_date(date(2025, 1, 5)) == "2024-25"

    def test_may_is_same_season(self):
        assert season_label_from_date(date(2025, 5, 25)) == "2024-25"

    def test_july_is_previous_season(self):
        assert season_label_from_date(date(2025, 7, 15)) == "2024-25"


class TestIsNoCrowd:
    def test_covid_match(self):
        assert is_no_crowd(date(2020, 9, 12)) is True

    def test_pre_covid(self):
        assert is_no_crowd(date(2020, 3, 1)) is False

    def test_post_covid(self):
        assert is_no_crowd(date(2021, 8, 14)) is False

    def test_boundary_start(self):
        assert is_no_crowd(date(2020, 6, 17)) is True

    def test_boundary_end(self):
        assert is_no_crowd(date(2021, 5, 17)) is True
