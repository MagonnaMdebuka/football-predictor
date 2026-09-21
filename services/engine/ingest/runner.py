"""Wiring layer connecting CLI commands to ingest components."""

import logging

from services.engine.ingest.config import IngestConfig
from services.engine.ingest.db_session import get_session

logger = logging.getLogger(__name__)


def run_seed() -> None:
    """Seed leagues, seasons, and team aliases."""
    from services.engine.ingest.seed import run_all_seeds

    config = IngestConfig()
    with get_session(config) as session:
        run_all_seeds(session)


def run_csv_backfill() -> None:
    """Backfill all season CSVs from football-data.co.uk."""
    from services.engine.ingest.csv_loader import backfill_all_seasons

    config = IngestConfig()
    with get_session(config) as session:
        backfill_all_seasons(session, config)


def run_fixtures_sync() -> None:
    """Sync upcoming fixtures from football-data.org."""
    from services.engine.ingest.fixtures_loader import sync_fixtures

    config = IngestConfig()
    with get_session(config) as session:
        sync_fixtures(session, config)


def run_verify_ingest() -> bool:
    """Run quality checks on all ingested data. Returns True if all pass."""
    from services.engine.ingest.quality import verify_all_seasons

    config = IngestConfig()
    with get_session(config) as session:
        reports = verify_all_seasons(session)

    all_passed = all(r.passed for r in reports)
    total = sum(r.match_count for r in reports)

    print(f"\n{'=' * 60}")
    print("INGEST VERIFICATION REPORT")
    print(f"{'=' * 60}")

    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.season_label}: {status} ({r.match_count} matches, {r.team_count} teams)")
        for err in r.errors:
            print(f"    ERROR: {err}")

    print(f"\nTotal matches: {total}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'=' * 60}")

    return all_passed


def run_aliases_review() -> None:
    """Show unconfirmed aliases for manual review."""
    from db.models import TeamAlias

    config = IngestConfig()
    with get_session(config) as session:
        unconfirmed = (
            session.query(TeamAlias)
            .filter_by(confirmed=False)
            .order_by(TeamAlias.source, TeamAlias.raw_name)
            .all()
        )

        if not unconfirmed:
            print("No unconfirmed aliases.")
            return

        print(f"\n{len(unconfirmed)} unconfirmed aliases:\n")
        for alias in unconfirmed:
            team_name = alias.team.canonical_name if alias.team else "UNLINKED"
            score = f"{alias.score:.1f}" if alias.score else "N/A"
            print(f"  [{alias.source}] '{alias.raw_name}' → '{team_name}' (score={score})")


def run_aliases_confirm(source: str, raw_name: str, team_name: str) -> None:
    """Manually confirm an alias mapping."""
    from db.models import Team, TeamAlias

    config = IngestConfig()
    with get_session(config) as session:
        team = session.query(Team).filter_by(canonical_name=team_name).first()
        if not team:
            print(f"Team not found: {team_name}")
            return

        alias = (
            session.query(TeamAlias)
            .filter_by(source=source, raw_name=raw_name)
            .first()
        )
        if not alias:
            print(f"Alias not found: [{source}] '{raw_name}'")
            return

        alias.team_id = team.id
        alias.confirmed = True
        alias.score = 100.0
        session.commit()
        print(f"Confirmed: [{source}] '{raw_name}' → '{team_name}'")
