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
    expected_teams: int = 20,
    expected_matches: int | None = None,
) -> QualityReport:
    """Run quality checks on a single season. Returns a QualityReport.

    Args:
        expected_teams: number of teams in the league (default 20).
        expected_matches: expected match count. Derived from expected_teams if not given.
    """
    if expected_matches is None:
        expected_matches = expected_teams * (expected_teams - 1)

    report = QualityReport(season_label=season_label)

    # Match count
    report.match_count = (
        session.query(func.count(Match.id))
        .filter_by(league_id=league_id, season_id=season_id)
        .scalar()
        or 0
    )

    # Team count (computed first so we can use it in match-count validation)
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

    # Self-consistent: actual matches == teams * (teams - 1) for a round-robin
    self_consistent_matches = report.team_count * (report.team_count - 1)
    self_consistent = (
        report.match_count == self_consistent_matches and report.team_count > 0
    )

    if not is_current:
        if report.match_count != expected_matches and not self_consistent:
            report.errors.append(
                f"Expected {expected_matches} matches, found {report.match_count}"
            )
        if report.team_count != expected_teams and not self_consistent:
            report.errors.append(
                f"Expected {expected_teams} teams, found {report.team_count}"
            )
    else:
        if report.match_count == 0:
            report.errors.append("No matches found for current season")

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


def verify_all_seasons(
    session: Session,
    league_code: str | None = None,
) -> list[QualityReport]:
    """Run quality checks on seasons. Returns list of QualityReports.

    Args:
        league_code: specific league code (e.g. "D1"). None = all active leagues.
    """
    from services.engine.config.league_defaults import LEAGUE_CONFIGS
    from services.engine.ingest.csv_loader import _is_current_season

    if league_code:
        leagues = session.query(League).filter_by(fd_couk_code=league_code, is_active=True).all()
    else:
        leagues = (
            session.query(League)
            .filter(League.fd_couk_code.isnot(None), League.is_active.is_(True))
            .all()
        )

    if not leagues:
        logger.error("No leagues found for verification")
        return []

    reports: list[QualityReport] = []

    for league in leagues:
        code = league.fd_couk_code
        lc = LEAGUE_CONFIGS.get(code)
        expected_teams = lc.num_teams if lc else 20

        seasons = (
            session.query(Season)
            .filter_by(league_id=league.id)
            .order_by(Season.label)
            .all()
        )

        logger.info("Verifying %s (%s) — %d seasons", league.name, code, len(seasons))

        total_matches = 0
        for season in seasons:
            is_current = _is_current_season(season.label)
            report = verify_season(
                session, league.id, season.id, season.label,
                is_current=is_current,
                expected_teams=expected_teams,
            )
            reports.append(report)
            total_matches += report.match_count

            status = "PASS" if report.passed else "FAIL"
            logger.info(
                "[%s] Season %s: %s (%d matches, %d teams)",
                code, season.label, status, report.match_count, report.team_count,
            )
            for err in report.errors:
                logger.error("  ERROR: %s", err)
            for warn in report.warnings:
                logger.warning("  WARN: %s", warn)

        logger.info("[%s] Total matches: %d", code, total_matches)

    return reports
