"""Tests for fixtures loader: JSON parsing, fixture upsert, merge with CSV match."""

from datetime import datetime, timezone

import pytest

from services.engine.ingest.fixtures_loader import _map_status, _parse_fd_org_match


class TestParseFdOrgMatch:
    def test_scheduled_match(self):
        data = {
            "id": 12345,
            "utcDate": "2024-08-17T14:00:00Z",
            "status": "TIMED",
            "homeTeam": {"name": "Arsenal FC"},
            "awayTeam": {"name": "Wolverhampton Wanderers FC"},
            "score": {"fullTime": {"home": None, "away": None}},
        }
        result = _parse_fd_org_match(data)

        assert result is not None
        assert result["home_team"] == "Arsenal FC"
        assert result["away_team"] == "Wolverhampton Wanderers FC"
        assert result["status"] == "scheduled"
        assert result["source_match_id"] == "12345"
        assert result["kickoff_time_known"] is True

    def test_finished_match(self):
        data = {
            "id": 12346,
            "utcDate": "2024-08-17T14:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"name": "Arsenal FC"},
            "awayTeam": {"name": "Chelsea FC"},
            "score": {
                "fullTime": {"home": 2, "away": 1},
                "halfTime": {"home": 1, "away": 0},
            },
        }
        result = _parse_fd_org_match(data)

        assert result["ft_home_goals"] == 2
        assert result["ft_away_goals"] == 1
        assert result["ht_home_goals"] == 1
        assert result["ht_away_goals"] == 0
        assert result["status"] == "finished"

    def test_missing_teams_returns_none(self):
        data = {
            "id": 12347,
            "utcDate": "2024-08-17T14:00:00Z",
            "status": "SCHEDULED",
            "homeTeam": {},
            "awayTeam": {},
            "score": {},
        }
        result = _parse_fd_org_match(data)
        assert result is None

    def test_missing_utc_date_returns_none(self):
        data = {
            "id": 12348,
            "status": "SCHEDULED",
            "homeTeam": {"name": "Arsenal FC"},
            "awayTeam": {"name": "Chelsea FC"},
        }
        result = _parse_fd_org_match(data)
        assert result is None

    def test_utc_date_parsed_correctly(self):
        data = {
            "id": 12349,
            "utcDate": "2024-12-26T15:00:00Z",
            "status": "TIMED",
            "homeTeam": {"name": "Liverpool FC"},
            "awayTeam": {"name": "Leicester City FC"},
            "score": {},
        }
        result = _parse_fd_org_match(data)
        assert result["kickoff_utc"].year == 2024
        assert result["kickoff_utc"].month == 12
        assert result["kickoff_utc"].day == 26
        assert result["kickoff_utc"].hour == 15


class TestMapStatus:
    def test_scheduled(self):
        assert _map_status("SCHEDULED") == "scheduled"

    def test_timed(self):
        assert _map_status("TIMED") == "scheduled"

    def test_finished(self):
        assert _map_status("FINISHED") == "finished"

    def test_in_play(self):
        assert _map_status("IN_PLAY") == "live"

    def test_postponed(self):
        assert _map_status("POSTPONED") == "postponed"

    def test_unknown(self):
        assert _map_status("UNKNOWN_STATUS") == "scheduled"
