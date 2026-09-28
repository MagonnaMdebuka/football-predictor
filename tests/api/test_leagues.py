"""Tests for the leagues API endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_league(id=1, name="Premier League", country="England",
                 fd_couk_code="E0", is_active=True,
                 ship_corners=False, ship_cards=False):
    """Create a mock League object."""
    lg = MagicMock()
    lg.id = id
    lg.name = name
    lg.country = country
    lg.fd_couk_code = fd_couk_code
    lg.is_active = is_active
    lg.ship_corners = ship_corners
    lg.ship_cards = ship_cards
    return lg


@pytest.mark.asyncio
async def test_list_leagues_returns_200(client, mock_db_session):
    """GET /api/v1/leagues returns 200 with league list."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        _make_league(),
    ]
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/leagues")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "Premier League"
    assert data[0]["code"] == "E0"


@pytest.mark.asyncio
async def test_list_leagues_empty(client, mock_db_session):
    """GET /api/v1/leagues returns empty list when no leagues."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/leagues")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_league_not_found(client, mock_db_session):
    """GET /api/v1/leagues/XX returns 404 for unknown league."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/leagues/XX")
    assert response.status_code == 404
