"""Seed leagues, seasons, and team aliases from CSV and constants."""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models import League, Season, Team, TeamAlias
from services.engine.ingest.constants import (
    PL_FD_COUK_CODE,
    PL_FD_ORG_CODE,
    SEASON_DATES,
)

logger = logging.getLogger(__name__)

SEEDS_DIR = Path(__file__).resolve().parents[3] / "db" / "seeds"


def seed_premier_league(session: Session) -> League:
    """Upsert the Premier League row and return it."""
    stmt = (
        insert(League)
        .values(
            name="Premier League",
            country="England",
            tier=1,
            is_active=True,
            fd_couk_code=PL_FD_COUK_CODE,
            fd_org_code=PL_FD_ORG_CODE,
            has_corners=True,
            has_cards=True,
            has_xg=False,
        )
        .on_conflict_do_nothing()
    )
    session.execute(stmt)
    session.flush()

    league = session.query(League).filter_by(name="Premier League").one()
    logger.info("Premier League seeded (id=%d)", league.id)
    return league


def seed_seasons(session: Session, league: League) -> dict[str, Season]:
    """Upsert seasons for the Premier League. Returns {label: Season}."""
    seasons: dict[str, Season] = {}
    for label, (start_str, end_str) in SEASON_DATES.items():
        start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        stmt = (
            insert(Season)
            .values(league_id=league.id, label=label, start_date=start, end_date=end)
            .on_conflict_do_nothing(index_elements=["league_id", "label"])
        )
        session.execute(stmt)

    session.flush()

    for s in session.query(Season).filter_by(league_id=league.id).all():
        seasons[s.label] = s

    logger.info("Seeded %d seasons", len(seasons))
    return seasons


def seed_teams_and_aliases(session: Session) -> int:
    """Seed teams and aliases from db/seeds/team_aliases.csv. Returns count of aliases seeded."""
    csv_path = SEEDS_DIR / "team_aliases.csv"
    if not csv_path.exists():
        logger.warning("Seed file not found: %s", csv_path)
        return 0

    count = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = row["canonical_name"].strip()
            country = row["country"].strip()
            source = row["source"].strip()
            raw_name = row["raw_name"].strip()

            # Upsert team
            stmt = (
                insert(Team)
                .values(canonical_name=canonical, country=country)
                .on_conflict_do_nothing()
            )
            session.execute(stmt)
            session.flush()

            team = session.query(Team).filter_by(canonical_name=canonical).one()

            # Upsert alias
            stmt = (
                insert(TeamAlias)
                .values(
                    team_id=team.id,
                    source=source,
                    raw_name=raw_name,
                    score=100.0,
                    confirmed=True,
                )
                .on_conflict_do_update(
                    index_elements=["source", "raw_name"],
                    set_={"team_id": team.id, "score": 100.0, "confirmed": True},
                )
            )
            session.execute(stmt)
            count += 1

    session.flush()
    logger.info("Seeded %d team aliases", count)
    return count


def run_all_seeds(session: Session) -> None:
    """Run all seed operations."""
    league = seed_premier_league(session)
    seed_seasons(session, league)
    seed_teams_and_aliases(session)
    session.commit()
    logger.info("All seeds complete")
