"""CSV backfill orchestrator: download → cache → parse → resolve → upsert.

Downloads season CSVs from football-data.co.uk, caches completed seasons locally,
resolves team names via the alias resolver, and upserts matches idempotently.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models import IngestRun, Match, MatchSourceRow, Referee
from services.engine.ingest.alias_resolver import AliasResolver
from services.engine.ingest.config import IngestConfig
from services.engine.ingest.constants import SEASON_CODES
from services.engine.ingest.csv_parser import compute_checksum, parse_csv

logger = logging.getLogger(__name__)


def _csv_url(config: IngestConfig, season_code: str, division: str = "E0") -> str:
    """Build the download URL for a football-data.co.uk CSV."""
    return f"{config.football_data_couk_base_url}/{season_code}/{division}.csv"


def _cache_path(config: IngestConfig, season_label: str, division: str = "E0") -> Path:
    """Return the local cache path for a season CSV."""
    raw_dir = Path(config.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{division}_{season_label.replace('-', '')}.csv"


def _download_csv(
    config: IngestConfig, season_label: str, season_code: str, division: str = "E0"
) -> Path:
    """Download a CSV if not already cached. Returns the local file path."""
    cache = _cache_path(config, season_label, division)

    # For completed seasons, use cache if it exists
    if cache.exists():
        logger.info("Using cached CSV: %s", cache)
        return cache

    url = _csv_url(config, season_code, division)
    logger.info("Downloading %s → %s", url, cache)

    response = httpx.get(url, follow_redirects=True, timeout=30.0, verify=False)
    response.raise_for_status()

    cache.write_bytes(response.content)
    return cache


def _is_current_season(season_label: str) -> bool:
    """Check if a season label refers to the current (incomplete) season."""
    start_year = int(season_label.split("-")[0])
    now = datetime.now(timezone.utc)
    # Current season if we're within its date range
    return start_year == now.year or (now.month < 8 and start_year == now.year - 1)


def _upsert_referee(session: Session, name: str | None) -> int | None:
    """Upsert a referee and return their ID."""
    if not name:
        return None

    stmt = insert(Referee).values(name=name).on_conflict_do_nothing()
    session.execute(stmt)
    session.flush()

    ref = session.query(Referee).filter_by(name=name).first()
    return ref.id if ref else None


def _upsert_match(
    session: Session,
    row: dict[str, Any],
    league_id: int,
    season_id: int,
    home_team_id: int,
    away_team_id: int,
    referee_id: int | None,
    checksum: str,
) -> bool:
    """Upsert a single match row. Returns True if a new row was written."""
    values = {
        "league_id": league_id,
        "season_id": season_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "referee_id": referee_id,
        "kickoff_utc": row["kickoff_utc"],
        "ft_home_goals": row["ft_home_goals"],
        "ft_away_goals": row["ft_away_goals"],
        "ht_home_goals": row.get("ht_home_goals"),
        "ht_away_goals": row.get("ht_away_goals"),
        "home_shots": row.get("home_shots"),
        "away_shots": row.get("away_shots"),
        "home_shots_on_target": row.get("home_shots_on_target"),
        "away_shots_on_target": row.get("away_shots_on_target"),
        "home_fouls": row.get("home_fouls"),
        "away_fouls": row.get("away_fouls"),
        "home_corners": row.get("home_corners"),
        "away_corners": row.get("away_corners"),
        "home_yellows": row.get("home_yellows"),
        "away_yellows": row.get("away_yellows"),
        "home_reds": row.get("home_reds"),
        "away_reds": row.get("away_reds"),
        "kickoff_time_known": row.get("kickoff_time_known", True),
        "no_crowd": row.get("no_crowd", False),
        "status": "finished" if row["ft_home_goals"] is not None else "scheduled",
    }

    stmt = (
        insert(Match)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["league_id", "season_id", "home_team_id", "away_team_id"],
            set_=values,
        )
    )
    result = session.execute(stmt)
    session.flush()

    # Get the match ID for the source row
    match = (
        session.query(Match)
        .filter_by(
            league_id=league_id,
            season_id=season_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        .one()
    )

    # Upsert source row
    src_stmt = (
        insert(MatchSourceRow)
        .values(
            match_id=match.id,
            source="fd_couk",
            raw=row.get("raw", {}),
            file_checksum=checksum,
        )
        .on_conflict_do_update(
            index_elements=["match_id", "source"],
            set_={"raw": row.get("raw", {}), "file_checksum": checksum},
        )
    )
    session.execute(src_stmt)

    return result.rowcount > 0


def load_season_csv(
    session: Session,
    config: IngestConfig,
    season_label: str,
    league_id: int,
    season_id: int,
    resolver: AliasResolver,
) -> tuple[int, int]:
    """Load a single season CSV. Returns (rows_written, rows_skipped)."""
    season_code = SEASON_CODES.get(season_label)
    if not season_code:
        logger.error("Unknown season: %s", season_label)
        return 0, 0

    # Download / use cache (re-download current season)
    cache = _cache_path(config, season_label)
    if _is_current_season(season_label) or not cache.exists():
        file_path = _download_csv(config, season_label, season_code)
    else:
        file_path = cache

    # Parse CSV
    rows = parse_csv(file_path)
    if not rows:
        return 0, 0

    checksum = compute_checksum(file_path)
    written = 0
    skipped = 0

    for row in rows:
        home_id = resolver.resolve("fd_couk", row["home_team"])
        away_id = resolver.resolve("fd_couk", row["away_team"])

        if home_id is None or away_id is None:
            skipped += 1
            continue

        referee_id = _upsert_referee(session, row.get("referee"))

        if _upsert_match(
            session, row, league_id, season_id, home_id, away_id, referee_id, checksum
        ):
            written += 1

    # Flush new aliases
    resolver.flush_new_aliases(session)
    session.flush()

    logger.info(
        "Season %s: %d written, %d skipped", season_label, written, skipped
    )
    return written, skipped


def backfill_all_seasons(session: Session, config: IngestConfig | None = None) -> None:
    """Backfill all configured seasons from football-data.co.uk."""
    if config is None:
        config = IngestConfig()

    from db.models import League, Season

    league = session.query(League).filter_by(fd_couk_code="E0").first()
    if not league:
        logger.error("Premier League not found — run seed first")
        return

    seasons = {s.label: s for s in session.query(Season).filter_by(league_id=league.id).all()}
    resolver = AliasResolver.from_session(session, config)

    total_written = 0
    total_skipped = 0

    for season_label in SEASON_CODES:
        season = seasons.get(season_label)
        if not season:
            logger.warning("Season %s not found in DB — skipping", season_label)
            continue

        # Create ingest run record
        run = IngestRun(source="fd_couk", job=f"csv_backfill_{season_label}", status="running")
        session.add(run)
        session.flush()

        try:
            written, skipped = load_season_csv(
                session, config, season_label, league.id, season.id, resolver
            )
            run.rows_written = written
            run.rows_skipped = skipped
            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc)
            total_written += written
            total_skipped += skipped
            session.commit()
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
            logger.error("Failed to load season %s: %s", season_label, exc)
            raise

    logger.info(
        "Backfill complete: %d written, %d skipped across %d seasons",
        total_written, total_skipped, len(SEASON_CODES),
    )
