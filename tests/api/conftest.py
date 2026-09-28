"""Shared test fixtures for API tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_db_engine():
    """Mock async database engine."""
    engine = AsyncMock()
    conn = AsyncMock()
    engine.connect.return_value.__aenter__ = AsyncMock(return_value=conn)
    engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock()
    return engine


@pytest.fixture
def mock_redis():
    """Mock async Redis client."""
    r = AsyncMock()
    r.ping = AsyncMock(return_value=True)
    return r


@pytest.fixture
def mock_db_session():
    """Mock async database session for dependency injection."""
    session = AsyncMock()
    return session


@pytest.fixture
async def client(mock_db_engine, mock_redis, mock_db_session):
    """Create test client with mocked dependencies."""
    with (
        patch("services.api.main.create_async_engine", return_value=mock_db_engine),
        patch("services.api.main.aioredis") as mock_aioredis,
    ):
        mock_aioredis.from_url.return_value = mock_redis

        from services.api.main import app
        from services.api.deps import get_db

        async def override_get_db():
            yield mock_db_session

        app.dependency_overrides[get_db] = override_get_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

        app.dependency_overrides.clear()
