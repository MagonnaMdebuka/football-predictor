"""Fixture sync from football-data.org v4 API.

Downloads upcoming and recently completed matches, resolves teams via aliases,
and upserts into the matches table. Merges with existing CSV-sourced match data
when the same match exists from both sources.
"""

import logging
from datetime import datetime, timedelta, timezone

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
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict]:
    """Fetch matches from football-data.org for the Premier League."""
    if date_from is None:
        date_from = datetime.now(timezone.utc) - timedelta(days=7)
    if date_to is None:
        date_to = datetime.now(timezone.utc) + timedelta(days=14)

    url = (
        f"{config.football_data_org_base_url}/competitions/{PL_FD_ORG_CODE}/matches"
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

    logger.info("Fetched %d fixtures from football-data.org", len(parsed))
    return parsed


def sync_fixtures(
    session: Session,
    config: IngestConfig | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> tuple[int, int]:
    """Sync fixtures from football-data.org. Returns (written, skipped)."""
    if config is None:
        config = IngestConfig()

    league = session.query(League).filter_by(fd_org_code=PL_FD_ORG_CODE).first()
    if not league:
        logger.error("Premier League not found — run seed first")
        return 0, 0

    seasons = {s.label: s for s in session.query(Season).filter_by(league_id=league.id).all()}
    resolver = AliasResolver.from_session(session, config)

    # Create ingest run
    run = IngestRun(source="fd_org", job="fixtures_sync", status="running")
    session.add(run)
    session.flush()

    client = RateLimitedClient(config, source_name="football-data.org")
    try:
        fixtures = fetch_fixtures(client, config, date_from, date_to)

        written = 0
        skipped = 0

        for fixture in fixtures:
            home_id = resolver.resolve("fd_org", fixture["home_team"])
            away_id = resolver.resolve("fd_org", fixture["away_team"])

            if home_id is None or away_id is None:
                skipped += 1
                continue

            # Determine season
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

            # Include scores for finished matches
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

            # Store source row
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

            src_stmt = (
                insert(MatchSourceRow)
                .values(
                    match_id=match.id,
                    source="fd_org",
                    raw=fixture,
                )
                .on_conflict_do_update(
                    index_elements=["match_id", "source"],
                    set_={"raw": fixture},
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

        logger.info("Fixture sync complete: %d written, %d skipped", written, skipped)
        return written, skipped

    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.finished_at = datetime.now(timezone.utc)
        session.commit()
        logger.error("Fixture sync failed: %s", exc)
        raise
    finally:
        client.close()
