"""Per-season data quality validation.

Checks: match count, team count, stat consistency, no negative values,
null rate thresholds.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from db.models import League, Match, Season

logger = logging.getLogger(__name__)

EXPECTED_MATCHES_PER_SEASON = 380
EXPECTED_TEAMS_PER_SEASON = 20
MAX_NULL_RATE = 0.05  # 5% null rate threshold for stat columns


@dataclass
class QualityReport:
    season_label: str
    match_count: int = 0
    team_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def verify_season(
    session: Session,
    league_id: int,
    season_id: int,
    season_label: str,
    is_current: bool = False,
) -> QualityReport:
    """Run quality checks on a single season. Returns a QualityReport."""
    report = QualityReport(season_label=season_label)

    # Match count
    report.match_count = (
        session.query(func.count(Match.id))
        .filter_by(league_id=league_id, season_id=season_id)
        .scalar()
        or 0
    )

    if not is_current:
        if report.match_count != EXPECTED_MATCHES_PER_SEASON:
            report.errors.append(
                f"Expected {EXPECTED_MATCHES_PER_SEASON} matches, found {report.match_count}"
            )
    else:
        if report.match_count == 0:
            report.errors.append("No matches found for current season")

    # Team count
    home_teams = (
        session.query(distinct(Match.home_team_id))
        .filter_by(league_id=league_id, season_id=season_id)
        .count()
    )
    away_teams = (
        session.query(distinct(Match.away_team_id))
        .filter_by(league_id=league_id, season_id=season_id)
        .count()
    )
    report.team_count = max(home_teams, away_teams)

    if not is_current:
        if report.team_count != EXPECTED_TEAMS_PER_SEASON:
            report.errors.append(
                f"Expected {EXPECTED_TEAMS_PER_SEASON} teams, found {report.team_count}"
            )

    # Stat consistency: shots on target <= shots
    if report.match_count > 0:
        bad_shots = (
            session.query(func.count(Match.id))
            .filter(
                Match.league_id == league_id,
                Match.season_id == season_id,
                Match.home_shots.isnot(None),
                Match.home_shots_on_target.isnot(None),
                Match.home_shots_on_target > Match.home_shots,
            )
            .scalar()
            or 0
        )
        bad_shots += (
            session.query(func.count(Match.id))
            .filter(
                Match.league_id == league_id,
                Match.season_id == season_id,
                Match.away_shots.isnot(None),
                Match.away_shots_on_target.isnot(None),
                Match.away_shots_on_target > Match.away_shots,
            )
            .scalar()
            or 0
        )
        if bad_shots > 0:
            report.errors.append(f"{bad_shots} matches with shots_on_target > shots")

    # No negative values in stat columns
    stat_cols = [
        Match.ft_home_goals, Match.ft_away_goals,
        Match.ht_home_goals, Match.ht_away_goals,
        Match.home_shots, Match.away_shots,
        Match.home_corners, Match.away_corners,
        Match.home_yellows, Match.away_yellows,
        Match.home_reds, Match.away_reds,
    ]
    for col in stat_cols:
        neg_count = (
            session.query(func.count(Match.id))
            .filter(
                Match.league_id == league_id,
                Match.season_id == season_id,
                col.isnot(None),
                col < 0,
            )
            .scalar()
            or 0
        )
        if neg_count > 0:
            report.errors.append(f"{neg_count} negative values in {col.key}")

    # Null rate for finished matches
    finished_count = (
        session.query(func.count(Match.id))
        .filter_by(league_id=league_id, season_id=season_id, status="finished")
        .scalar()
        or 0
    )

    if finished_count > 0:
        for col in [Match.ft_home_goals, Match.ft_away_goals]:
            null_count = (
                session.query(func.count(Match.id))
                .filter(
                    Match.league_id == league_id,
                    Match.season_id == season_id,
                    Match.status == "finished",
                    col.is_(None),
                )
                .scalar()
                or 0
            )
            null_rate = null_count / finished_count
            if null_rate > MAX_NULL_RATE:
                report.errors.append(
                    f"{col.key} null rate {null_rate:.1%} exceeds threshold {MAX_NULL_RATE:.1%}"
                )

    return report


def verify_all_seasons(session: Session) -> list[QualityReport]:
    """Run quality checks on all seasons. Returns list of QualityReports."""
    from services.engine.ingest.csv_loader import _is_current_season

    league = session.query(League).filter_by(fd_couk_code="E0").first()
    if not league:
        logger.error("Premier League not found")
        return []

    seasons = session.query(Season).filter_by(league_id=league.id).order_by(Season.label).all()
    reports: list[QualityReport] = []

    total_matches = 0
    for season in seasons:
        is_current = _is_current_season(season.label)
        report = verify_season(session, league.id, season.id, season.label, is_current)
        reports.append(report)
        total_matches += report.match_count

        status = "PASS" if report.passed else "FAIL"
        logger.info(
            "Season %s: %s (%d matches, %d teams)",
            season.label, status, report.match_count, report.team_count,
        )
        for err in report.errors:
            logger.error("  ERROR: %s", err)
        for warn in report.warnings:
            logger.warning("  WARN: %s", warn)

    logger.info("Total matches across all seasons: %d", total_matches)
    return reports
