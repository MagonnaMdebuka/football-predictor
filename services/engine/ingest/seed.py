"""Seed leagues, seasons, and team aliases from CSV and constants."""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models import League, Season, Team, TeamAlias
from services.engine.ingest.constants import (
    BL_FD_COUK_CODE,
    BL_FD_ORG_CODE,
    CH_FD_COUK_CODE,
    CH_FD_ORG_CODE,
    L1_FD_COUK_CODE,
    L1_FD_ORG_CODE,
    LL_FD_COUK_CODE,
    LL_FD_ORG_CODE,
    PL_FD_COUK_CODE,
    PL_FD_ORG_CODE,
    SA_FD_COUK_CODE,
    SA_FD_ORG_CODE,
    SEASON_DATES,
)

logger = logging.getLogger(__name__)

SEEDS_DIR = Path(__file__).resolve().parents[3] / "db" / "seeds"


def seed_league(
    session: Session,
    *,
    name: str,
    country: str,
    tier: int,
    fd_couk_code: str,
    fd_org_code: int,
    has_corners: bool = False,
    has_cards: bool = False,
    ship_corners: bool = False,
    ship_cards: bool = False,
    has_xg: bool = False,
) -> League:
    """Upsert a league row and return it.

    Uses fd_couk_code as the natural key for idempotent inserts.
    """
    existing = session.query(League).filter_by(fd_couk_code=fd_couk_code).first()
    if existing:
        logger.info("%s already seeded (id=%d)", name, existing.id)
        return existing

    league = League(
        name=name,
        country=country,
        tier=tier,
        is_active=True,
        fd_couk_code=fd_couk_code,
        fd_org_code=fd_org_code,
        has_corners=has_corners,
        has_cards=has_cards,
        ship_corners=ship_corners,
        ship_cards=ship_cards,
        has_xg=has_xg,
    )
    session.add(league)
    session.flush()
    logger.info("%s seeded (id=%d)", name, league.id)
    return league


def seed_premier_league(session: Session) -> League:
    """Upsert the Premier League row and return it."""
    return seed_league(
        session,
        name="Premier League",
        country="England",
        tier=1,
        fd_couk_code=PL_FD_COUK_CODE,
        fd_org_code=PL_FD_ORG_CODE,
        has_corners=True,
        has_cards=True,
    )


def seed_bundesliga(session: Session) -> League:
    """Upsert the Bundesliga row and return it."""
    return seed_league(
        session,
        name="Bundesliga",
        country="Germany",
        tier=1,
        fd_couk_code=BL_FD_COUK_CODE,
        fd_org_code=BL_FD_ORG_CODE,
        has_corners=True,
        has_cards=True,
    )


def seed_la_liga(session: Session) -> League:
    """Upsert the La Liga row and return it."""
    return seed_league(
        session,
        name="La Liga",
        country="Spain",
        tier=1,
        fd_couk_code=LL_FD_COUK_CODE,
        fd_org_code=LL_FD_ORG_CODE,
        has_corners=True,
        has_cards=True,
    )


def seed_serie_a(session: Session) -> League:
    """Upsert the Serie A row and return it."""
    return seed_league(
        session,
        name="Serie A",
        country="Italy",
        tier=1,
        fd_couk_code=SA_FD_COUK_CODE,
        fd_org_code=SA_FD_ORG_CODE,
        has_corners=True,
        has_cards=True,
    )


def seed_ligue_1(session: Session) -> League:
    """Upsert the Ligue 1 row and return it."""
    return seed_league(
        session,
        name="Ligue 1",
        country="France",
        tier=1,
        fd_couk_code=L1_FD_COUK_CODE,
        fd_org_code=L1_FD_ORG_CODE,
        has_corners=True,
        has_cards=True,
    )


def seed_championship(session: Session) -> League:
    """Upsert the Championship row and return it."""
    return seed_league(
        session,
        name="Championship",
        country="England",
        tier=2,
        fd_couk_code=CH_FD_COUK_CODE,
        fd_org_code=CH_FD_ORG_CODE,
        has_corners=True,
        has_cards=True,
    )


def seed_seasons(session: Session, league: League) -> dict[str, Season]:
    """Upsert seasons for a league. Returns {label: Season}."""
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
    pl = seed_premier_league(session)
    seed_seasons(session, pl)

    bl = seed_bundesliga(session)
    seed_seasons(session, bl)

    ll = seed_la_liga(session)
    seed_seasons(session, ll)

    sa = seed_serie_a(session)
    seed_seasons(session, sa)

    l1 = seed_ligue_1(session)
    seed_seasons(session, l1)

    ch = seed_championship(session)
    seed_seasons(session, ch)

    seed_teams_and_aliases(session)
    session.commit()
    logger.info("All seeds complete")
