"""FastAPI dependency injection for database sessions."""

from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session from the app-level engine."""
    factory = async_sessionmaker(request.app.state.db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
