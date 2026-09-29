"""Fixture list endpoint."""

from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import League, Match, Team
from services.api.deps import get_db
from services.api.routers.leagues import _latest_prediction_summary
from services.api.schemas import FixtureOut

router = APIRouter(prefix="/api/v1/fixtures", tags=["fixtures"])


@router.get("/next-date")
async def next_fixture_date(
    db: AsyncSession = Depends(get_db),
) -> dict[str, str | None]:
    """Return the earliest date that has a scheduled fixture, or null."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.min(Match.kickoff_utc))
        .where(Match.status == "scheduled")
        .where(Match.kickoff_utc >= today)
    )
    earliest = result.scalar_one_or_none()
    if earliest is None:
        return {"date": None}
    return {"date": earliest.strftime("%Y-%m-%d")}


@router.get("", response_model=list[FixtureOut])
async def list_fixtures(
    date: date | None = Query(default=None, description="ISO date (default: today)"),
    league: str | None = Query(default=None, description="League code filter"),
    db: AsyncSession = Depends(get_db),
):
    """List fixtures for a given date with their latest predictions."""
    target_date = date or datetime.now(timezone.utc).date()

    query = (
        select(Match)
        .where(
            Match.kickoff_utc >= datetime(target_date.year, target_date.month,
                                          target_date.day, tzinfo=timezone.utc),
            Match.kickoff_utc < datetime(target_date.year, target_date.month,
                                         target_date.day, tzinfo=timezone.utc).replace(
                                             hour=23, minute=59, second=59),
        )
        .order_by(Match.kickoff_utc)
    )

    if league:
        league_result = await db.execute(
            select(League.id).where(League.fd_couk_code == league)
        )
        league_id = league_result.scalar_one_or_none()
        if league_id is not None:
            query = query.where(Match.league_id == league_id)

    result = await db.execute(query)
    matches = result.scalars().all()

    fixtures = []
    for m in matches:
        home = await db.get(Team, m.home_team_id)
        away = await db.get(Team, m.away_team_id)

        # Get league info
        lg = await db.get(League, m.league_id)
        league_code = lg.fd_couk_code if lg else None
        league_name = lg.name if lg else None

        pred_summary = await _latest_prediction_summary(db, m.id)

        fixtures.append(FixtureOut(
            id=m.id,
            home_team=home.canonical_name if home else "Unknown",
            away_team=away.canonical_name if away else "Unknown",
            kickoff_utc=m.kickoff_utc,
            status=m.status,
            league_code=league_code,
            league_name=league_name,
            ft_home_goals=m.ft_home_goals,
            ft_away_goals=m.ft_away_goals,
            prediction=pred_summary,
        ))

    return fixtures
