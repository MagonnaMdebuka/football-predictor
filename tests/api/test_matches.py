"""Tests for the match detail API endpoint."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_match(id=1, home_team_id=1, away_team_id=2, league_id=1,
                status="scheduled", referee_id=None):
    m = MagicMock()
    m.id = id
    m.home_team_id = home_team_id
    m.away_team_id = away_team_id
    m.league_id = league_id
    m.status = status
    m.kickoff_utc = datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc)
    m.referee_id = referee_id
    return m


def _make_team(id=1, name="Arsenal"):
    t = MagicMock()
    t.id = id
    t.canonical_name = name
    return t


@pytest.mark.asyncio
async def test_match_not_found(client, mock_db_session):
    """GET /api/v1/matches/999 returns 404 when match does not exist."""
    mock_db_session.get = AsyncMock(return_value=None)

    response = await client.get("/api/v1/matches/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_match_returns_200(client, mock_db_session):
    """GET /api/v1/matches/1 returns 200 with match detail."""
    match = _make_match()
    home = _make_team(1, "Arsenal")
    away = _make_team(2, "Chelsea")

    # db.get calls: first for match, then for teams
    async def mock_get(model, id):
        if id == 1 and model is not MagicMock:
            return match
        return None

    mock_db_session.get = AsyncMock(side_effect=[match, home, away])

    # Prediction query returns None (no predictions yet)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/matches/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["home_team"] == "Arsenal"
    assert data["away_team"] == "Chelsea"
    assert data["prediction"] is None
    assert data["markets"] == {}


@pytest.mark.asyncio
async def test_match_response_shape(client, mock_db_session):
    """Response includes expected top-level keys."""
    match = _make_match()
    home = _make_team(1, "Arsenal")
    away = _make_team(2, "Chelsea")

    mock_db_session.get = AsyncMock(side_effect=[match, home, away])

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/matches/1")
    data = response.json()
    expected_keys = {"id", "home_team", "away_team", "kickoff_utc",
                     "status", "referee", "prediction", "markets", "grid"}
    assert set(data.keys()) == expected_keys
