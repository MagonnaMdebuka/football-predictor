"""Fixture sync from football-data.org v4 API.

Downloads upcoming and recently completed matches, resolves teams via aliases,
and upserts into the matches table. Merges with existing CSV-sourced match data
when the same match exists from both sources.

Offline mode reads the same JSON format from data/fixtures/ when the API is
unreachable (e.g. behind FortiClient).
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models import IngestRun, League, Match, MatchSourceRow, Season
from services.engine.ingest.alias_resolver import AliasResolver
from services.engine.ingest.config import IngestConfig
from services.engine.ingest.constants import (
    FD_ORG_STATUS_FINISHED,
    FD_ORG_STATUS_SCHEDULED,
    FD_ORG_STATUS_TIMED,
    PL_FD_ORG_CODE,
)
from services.engine.ingest.http_client import RateLimitedClient
from services.engine.ingest.normalise import season_label_from_date

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "data" / "fixtures"

# football-data.org competition code (in filenames) → our fd_couk league code
FD_ORG_FILE_PREFIX_TO_LEAGUE: dict[str, str] = {
    "PL": "E0",
    "BL1": "D1",
    "PD": "SP1",
    "SA": "I1",
    "FL1": "F1",
    "ELC": "E1",
}


def _parse_fd_org_match(match_data: dict) -> dict | None:
    """Parse a single match object from football-data.org v4 response."""
    try:
        utc_date = match_data.get("utcDate")
        if not utc_date:
            return None

        kickoff = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
        status = match_data.get("status", "SCHEDULED")

        home_team = match_data.get("homeTeam", {}).get("name")
        away_team = match_data.get("awayTeam", {}).get("name")
        if not home_team or not away_team:
            return None

        result: dict = {
            "kickoff_utc": kickoff,
            "home_team": home_team,
            "away_team": away_team,
            "status": _map_status(status),
            "source_match_id": str(match_data.get("id", "")),
            "kickoff_time_known": status in (FD_ORG_STATUS_TIMED, FD_ORG_STATUS_FINISHED),
        }

        score = match_data.get("score", {})
        if status == FD_ORG_STATUS_FINISHED:
            full_time = score.get("fullTime", {})
            half_time = score.get("halfTime", {})
            result["ft_home_goals"] = full_time.get("home")
            result["ft_away_goals"] = full_time.get("away")
            result["ht_home_goals"] = half_time.get("home")
            result["ht_away_goals"] = half_time.get("away")

        return result
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse fd.org match: %s", exc)
        return None


def _map_status(fd_org_status: str) -> str:
    """Map football-data.org status to our internal status."""
    mapping = {
        "SCHEDULED": "scheduled",
        "TIMED": "scheduled",
        "IN_PLAY": "live",
        "PAUSED": "live",
        "FINISHED": "finished",
        "POSTPONED": "postponed",
        "CANCELLED": "cancelled",
        "SUSPENDED": "suspended",
    }
    return mapping.get(fd_org_status, "scheduled")


def fetch_fixtures(
    client: RateLimitedClient,
    config: IngestConfig,
    competition_code: int = PL_FD_ORG_CODE,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict]:
    """Fetch matches from football-data.org for a competition."""
    if date_from is None:
        date_from = datetime.now(timezone.utc) - timedelta(days=7)
    if date_to is None:
        date_to = datetime.now(timezone.utc) + timedelta(days=14)

    url = (
        f"{config.football_data_org_base_url}/competitions/{competition_code}/matches"
        f"?dateFrom={date_from.strftime('%Y-%m-%d')}"
        f"&dateTo={date_to.strftime('%Y-%m-%d')}"
    )

    headers = {"X-Auth-Token": config.football_data_org_token}
    response = client.get(url, headers=headers)
    data = response.json()

    matches_data = data.get("matches", [])
    parsed = []
    for m in matches_data:
        row = _parse_fd_org_match(m)
        if row:
            parsed.append(row)

    logger.info("Fetched %d fixtures from football-data.org (comp=%d)", len(parsed), competition_code)
    return parsed


def load_offline_fixtures(league_code: str | None = None) -> dict[str, list[dict]]:
    """Load fixtures from JSON files in data/fixtures/.

    Returns {fd_couk_code: [parsed_fixture_dicts]}.
    """
    result: dict[str, list[dict]] = {}

    for prefix, code in FD_ORG_FILE_PREFIX_TO_LEAGUE.items():
        if league_code and code != league_code:
            continue

        path = FIXTURES_DIR / f"{prefix}_fixtures.json"
        if not path.exists():
            logger.warning("Offline fixture file not found: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        matches_data = data.get("matches", [])
        parsed = []
        for m in matches_data:
            row = _parse_fd_org_match(m)
            if row:
                parsed.append(row)

        logger.info("Loaded %d fixtures from %s (%s)", len(parsed), path.name, code)
        result[code] = parsed

    return result


def sync_fixtures(
    session: Session,
    config: IngestConfig | None = None,
    league_code: str | None = None,
    offline: bool = False,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> tuple[int, int]:
    """Sync fixtures from football-data.org. Returns (written, skipped).

    Args:
        league_code: fd_couk_code (e.g. "E0"). If None, syncs all active leagues.
        offline: read from JSON files in data/fixtures/ instead of calling the API.
    """
    if config is None:
        config = IngestConfig()

    if offline:
        return _sync_offline(session, config, league_code)

    if league_code is None:
        return _sync_all_leagues(session, config, date_from, date_to)

    league = (
        session.query(League)
        .filter_by(fd_couk_code=league_code, is_active=True)
        .first()
    )
    if not league:
        logger.error("League %s not found — run seed first", league_code)
        return 0, 0
    if not league.fd_org_code:
        logger.error("League %s has no fd_org_code", league_code)
        return 0, 0

    return _sync_single_league(session, config, league, date_from, date_to)


def _sync_offline(
    session: Session,
    config: IngestConfig,
    league_code: str | None = None,
) -> tuple[int, int]:
    """Sync fixtures from offline JSON files. Returns total (written, skipped)."""
    all_fixtures = load_offline_fixtures(league_code)
    if not all_fixtures:
        logger.error("No offline fixture files found")
        return 0, 0

    total_written = 0
    total_skipped = 0

    for code, fixtures in all_fixtures.items():
        league = (
            session.query(League)
            .filter_by(fd_couk_code=code, is_active=True)
            .first()
        )
        if not league:
            logger.warning("League %s not found — skipping", code)
            continue

        w, s = _upsert_fixtures(session, config, league, fixtures)
        total_written += w
        total_skipped += s
        logger.info("[%s] %s: %d written, %d skipped", code, league.name, w, s)

    logger.info(
        "Offline sync complete: %d written, %d skipped across %d leagues",
        total_written, total_skipped, len(all_fixtures),
    )
    return total_written, total_skipped


def _sync_all_leagues(
    session: Session,
    config: IngestConfig,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> tuple[int, int]:
    """Sync fixtures for all active leagues. Returns total (written, skipped)."""
    leagues = (
        session.query(League)
        .filter(League.fd_org_code.isnot(None), League.is_active.is_(True))
        .order_by(League.fd_couk_code)
        .all()
    )
    if not leagues:
        logger.error("No active leagues with fd_org_code found")
        return 0, 0

    total_written = 0
    total_skipped = 0
    for league in leagues:
        logger.info("Syncing fixtures for %s (%s)...", league.name, league.fd_couk_code)
        w, s = _sync_single_league(session, config, league, date_from, date_to)
        total_written += w
        total_skipped += s

    logger.info(
        "All leagues synced: %d written, %d skipped across %d leagues",
        total_written, total_skipped, len(leagues),
    )
    return total_written, total_skipped


def _upsert_fixtures(
    session: Session,
    config: IngestConfig,
    league: League,
    fixtures: list[dict],
) -> tuple[int, int]:
    """Upsert parsed fixtures for a league. Returns (written, skipped)."""
    seasons = {s.label: s for s in session.query(Season).filter_by(league_id=league.id).all()}
    resolver = AliasResolver.from_session(session, config)

    run = IngestRun(source="fd_org", job="fixtures_sync", status="running")
    session.add(run)
    session.flush()

    try:
        written = 0
        skipped = 0

        for fixture in fixtures:
            home_id = resolver.resolve("fd_org", fixture["home_team"])
            away_id = resolver.resolve("fd_org", fixture["away_team"])

            if home_id is None or away_id is None:
                skipped += 1
                continue

            match_date = fixture["kickoff_utc"].date()
            season_label = season_label_from_date(match_date)
            season = seasons.get(season_label)
            if not season:
                logger.warning("No season found for %s (date %s)", season_label, match_date)
                skipped += 1
                continue

            values = {
                "league_id": league.id,
                "season_id": season.id,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "kickoff_utc": fixture["kickoff_utc"],
                "status": fixture["status"],
                "source_match_id": fixture.get("source_match_id"),
                "kickoff_time_known": fixture.get("kickoff_time_known", True),
            }

            if fixture["status"] == "finished":
                values["ft_home_goals"] = fixture.get("ft_home_goals")
                values["ft_away_goals"] = fixture.get("ft_away_goals")
                values["ht_home_goals"] = fixture.get("ht_home_goals")
                values["ht_away_goals"] = fixture.get("ht_away_goals")

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

            match = (
                session.query(Match)
                .filter_by(
                    league_id=league.id,
                    season_id=season.id,
                    home_team_id=home_id,
                    away_team_id=away_id,
                )
                .one()
            )

            raw_serialisable = {
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in fixture.items()
            }
            src_stmt = (
                insert(MatchSourceRow)
                .values(
                    match_id=match.id,
                    source="fd_org",
                    raw=raw_serialisable,
                )
                .on_conflict_do_update(
                    index_elements=["match_id", "source"],
                    set_={"raw": raw_serialisable},
                )
            )
            session.execute(src_stmt)

            if result.rowcount > 0:
                written += 1

        resolver.flush_new_aliases(session)

        run.rows_written = written
        run.rows_skipped = skipped
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        session.commit()

        return written, skipped

    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.finished_at = datetime.now(timezone.utc)
        session.commit()
        logger.error("Fixture sync failed for %s: %s", league.name, exc)
        raise


def _sync_single_league(
    session: Session,
    config: IngestConfig,
    league: League,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> tuple[int, int]:
    """Sync fixtures for a single league via API. Returns (written, skipped)."""
    client = RateLimitedClient(config, source_name="football-data.org")
    try:
        fixtures = fetch_fixtures(
            client, config, competition_code=league.fd_org_code,
            date_from=date_from, date_to=date_to,
        )
        written, skipped = _upsert_fixtures(session, config, league, fixtures)
        logger.info(
            "Fixture sync complete for %s: %d written, %d skipped",
            league.name, written, skipped,
        )
        return written, skipped
    finally:
        client.close()
