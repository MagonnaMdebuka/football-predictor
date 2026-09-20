"""Test the /health endpoint via ASGI."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_db_engine():
    engine = AsyncMock()
    conn = AsyncMock()
    engine.connect.return_value.__aenter__ = AsyncMock(return_value=conn)
    engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock()
    return engine


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.ping = AsyncMock(return_value=True)
    return r


@pytest.mark.asyncio
async def test_health_returns_200(mock_db_engine, mock_redis):
    with (
        patch("services.api.main.create_async_engine", return_value=mock_db_engine),
        patch("services.api.main.aioredis") as mock_aioredis,
    ):
        mock_aioredis.from_url.return_value = mock_redis

        from services.api.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "api"
    assert "timestamp" in data
    assert data["checks"]["database"] == "ok"
    assert data["checks"]["redis"] == "ok"
