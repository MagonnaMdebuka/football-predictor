"""Shared CLI helpers for multi-league iteration."""

from __future__ import annotations

import logging

from sqlalchemy import select

from db.models import League
from services.engine.ingest.db_session import get_session

logger = logging.getLogger(__name__)


def get_active_league_codes() -> list[str]:
    """Query the database for leagues where is_active = true.

    Returns a list of fd_couk_code strings (e.g. ["E0", "E1", "SP1"]).
    Adding a league is a database insert — no code changes needed.
    """
    with get_session() as session:
        codes = session.execute(
            select(League.fd_couk_code)
            .where(League.is_active.is_(True))
            .where(League.fd_couk_code.isnot(None))
            .order_by(League.fd_couk_code)
        ).scalars().all()

    if not codes:
        logger.warning("No active leagues found in the database")

    return list(codes)
