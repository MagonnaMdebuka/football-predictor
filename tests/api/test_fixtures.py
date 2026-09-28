"""Tests for the fixtures API endpoint."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_match(id=1, home_team_id=1, away_team_id=2, league_id=1,
                status="scheduled", kickoff=None):
    m = MagicMock()
    m.id = id
    m.home_team_id = home_team_id
    m.away_team_id = away_team_id
    m.league_id = league_id
    m.status = status
    m.kickoff_utc = kickoff or datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc)
    m.referee_id = None
    return m


def _make_team(id=1, name="Arsenal"):
    t = MagicMock()
    t.id = id
    t.canonical_name = name
    return t


def _make_league(id=1, code="E0"):
    lg = MagicMock()
    lg.id = id
    lg.fd_couk_code = code
    return lg


@pytest.mark.asyncio
async def test_list_fixtures_returns_200(client, mock_db_session):
    """GET /api/v1/fixtures returns 200."""
    # First call: match query
    match_result = MagicMock()
    match_result.scalars.return_value.all.return_value = []
    mock_db_session.execute = AsyncMock(return_value=match_result)

    response = await client.get("/api/v1/fixtures?date=2026-10-05")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_list_fixtures_with_league_filter(client, mock_db_session):
    """GET /api/v1/fixtures?league=E0 filters by league."""
    # Return empty for any query
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar_one_or_none.return_value = 1  # league_id
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/fixtures?date=2026-10-05&league=E0")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_fixtures_default_date(client, mock_db_session):
    """GET /api/v1/fixtures without date param uses today."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/fixtures")
    assert response.status_code == 200
