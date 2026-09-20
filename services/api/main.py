from contextlib import asynccontextmanager
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from services.api.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    yield
    await app.state.redis.aclose()
    await app.state.db_engine.dispose()


app = FastAPI(title="Football Predictor API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    db_ok = await _check_db(app.state.db_engine)
    redis_ok = await _check_redis(app.state.redis)
    healthy = db_ok and redis_ok
    return {
        "status": "healthy" if healthy else "degraded",
        "service": "api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "database": "ok" if db_ok else "error",
            "redis": "ok" if redis_ok else "error",
        },
    }


async def _check_db(engine: AsyncEngine) -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _check_redis(r: aioredis.Redis) -> bool:
    try:
        return await r.ping()
    except Exception:
        return False
