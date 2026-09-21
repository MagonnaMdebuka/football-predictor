"""Synchronous SQLAlchemy session factory for the ingest worker."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from services.engine.ingest.config import IngestConfig

_session_factory: sessionmaker | None = None


def get_session_factory(config: IngestConfig | None = None) -> sessionmaker:
    """Return a session factory, creating the engine on first call."""
    global _session_factory
    if _session_factory is None:
        if config is None:
            config = IngestConfig()
        engine = create_engine(config.database_url, echo=False)
        _session_factory = sessionmaker(bind=engine)
    return _session_factory


@contextmanager
def get_session(config: IngestConfig | None = None) -> Generator[Session, None, None]:
    """Context manager providing a transactional session scope."""
    factory = get_session_factory(config)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_session_factory() -> None:
    """Reset the session factory (useful for testing)."""
    global _session_factory
    _session_factory = None
