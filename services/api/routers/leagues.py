"""League endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import League, Match, Prediction, Season, Team
from services.api.deps import get_db
from services.api.schemas import (
    FixtureOut,
    LeagueDetailOut,
    LeagueOut,
    PredictionSummary,
    StandingsRowOut,
)
from services.engine.predict.entropy import compute_confidence
from services.engine.predict.utils import decompress_grid

router = APIRouter(prefix="/api/v1/leagues", tags=["leagues"])


@router.get("", response_model=list[LeagueOut])
async def list_leagues(db: AsyncSession = Depends(get_db)):
    """List all active leagues."""
    result = await db.execute(
        select(League).where(League.is_active == True).order_by(League.name)  # noqa: E712
    )
    leagues = result.scalars().all()
    return [
        LeagueOut(
            id=lg.id,
            name=lg.name,
            country=lg.country,
            code=lg.fd_couk_code or "",
            is_active=lg.is_active,
            ship_corners=lg.ship_corners,
            ship_cards=lg.ship_cards,
        )
        for lg in leagues
    ]


@router.get("/{code}", response_model=LeagueDetailOut)
async def get_league(code: str, db: AsyncSession = Depends(get_db)):
    """Get league details with fixtures and standings."""
    result = await db.execute(
        select(League).where(League.fd_couk_code == code)
    )
    league = result.scalar_one_or_none()
    if league is None:
        raise HTTPException(status_code=404, detail=f"League {code} not found")

    league_out = LeagueOut(
        id=league.id,
        name=league.name,
        country=league.country,
        code=league.fd_couk_code or "",
        is_active=league.is_active,
        ship_corners=league.ship_corners,
        ship_cards=league.ship_cards,
    )

    # Fixtures: scheduled matches with latest prediction
    fixtures_result = await db.execute(
        select(Match)
        .where(Match.league_id == league.id)
        .where(Match.status == "scheduled")
        .order_by(Match.kickoff_utc)
    )
    matches = fixtures_result.scalars().all()
    fixtures = []
    for m in matches:
        home = await db.get(Team, m.home_team_id)
        away = await db.get(Team, m.away_team_id)
        pred_summary = await _latest_prediction_summary(db, m.id)
        fixtures.append(FixtureOut(
            id=m.id,
            home_team=home.canonical_name if home else "Unknown",
            away_team=away.canonical_name if away else "Unknown",
            kickoff_utc=m.kickoff_utc,
            status=m.status,
            league_code=code,
            league_name=league.name,
            ft_home_goals=m.ft_home_goals,
            ft_away_goals=m.ft_away_goals,
            prediction=pred_summary,
        ))

    # Standings: from finished matches in the current season
    standings = await _compute_standings(db, league.id)

    return LeagueDetailOut(league=league_out, fixtures=fixtures, standings=standings)


def _grid_most_likely(grid: np.ndarray) -> tuple[int, int, float]:
    """Find most likely scoreline from an 11x11 grid array."""
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    return int(idx[0]), int(idx[1]), float(grid[idx])


def _grid_1x2(grid: np.ndarray) -> tuple[float, float, float]:
    """Compute home/draw/away probabilities from an 11x11 grid."""
    home = float(np.tril(grid, k=-1).sum())
    draw = float(np.trace(grid))
    away = float(np.triu(grid, k=1).sum())
    return home, draw, away


def _check_disagreement(grid: np.ndarray) -> str | None:
    """Check for scoreline disagreement using raw numpy array.

    Returns an explanatory message if the modal scoreline implies a different
    result from the 1X2 favourite, or None if they agree.
    """
    h_goals, a_goals, modal_prob = _grid_most_likely(grid)
    home_p, draw_p, away_p = _grid_1x2(grid)

    # Determine modal result
    if h_goals > a_goals:
        modal_result = "home"
    elif h_goals < a_goals:
        modal_result = "away"
    else:
        modal_result = "draw"

    # Determine 1X2 favourite
    best = max(home_p, draw_p, away_p)
    if best == home_p:
        fav_result = "home"
    elif best == away_p:
        fav_result = "away"
    else:
        fav_result = "draw"

    if modal_result == fav_result:
        return None

    score_str = f"{h_goals}-{a_goals}"
    return (
        f"Most likely score {score_str} ({modal_prob:.0%}) implies "
        f"{modal_result}, but overall probability favours "
        f"{fav_result} win ({best:.0%}). Many {fav_result}-win "
        f"scores collectively outweigh the single most likely "
        f"{modal_result}."
    )


async def _latest_prediction_summary(
    db: AsyncSession, match_id: int,
) -> PredictionSummary | None:
    """Get the latest prediction for a match as a summary."""
    result = await db.execute(
        select(Prediction)
        .where(Prediction.match_id == match_id)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    )
    pred = result.scalar_one_or_none()
    if pred is None:
        return None

    most_likely = None
    scoreline_note = None

    if pred.grid_compressed:
        grid_arr = decompress_grid(pred.grid_compressed)
        h, a, _ = _grid_most_likely(grid_arr)
        most_likely = f"{h}-{a}"
        scoreline_note = _check_disagreement(grid_arr)

    confidence = compute_confidence(
        pred.home_win_prob, pred.draw_prob, pred.away_win_prob,
    )

    return PredictionSummary(
        home_win_prob=pred.home_win_prob,
        draw_prob=pred.draw_prob,
        away_win_prob=pred.away_win_prob,
        home_expected_goals=pred.home_expected_goals,
        away_expected_goals=pred.away_expected_goals,
        most_likely_score=most_likely,
        confidence=confidence,
        scoreline_note=scoreline_note,
    )


async def _compute_standings(
    db: AsyncSession, league_id: int,
) -> list[StandingsRowOut]:
    """Compute league standings from finished matches in the latest season."""
    season_result = await db.execute(
        select(Season)
        .where(Season.league_id == league_id)
        .order_by(Season.start_date.desc())
        .limit(1)
    )
    season = season_result.scalar_one_or_none()
    if season is None:
        return []

    # Collect all team IDs from every match in the season (scheduled + finished)
    all_matches_result = await db.execute(
        select(Match)
        .where(Match.league_id == league_id)
        .where(Match.season_id == season.id)
    )
    all_matches = all_matches_result.scalars().all()

    # Initialise stats for every team seen in any match
    stats: dict[int, dict] = {}
    for m in all_matches:
        for tid in (m.home_team_id, m.away_team_id):
            if tid not in stats:
                stats[tid] = {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}

    # Update stats from finished matches only
    for m in all_matches:
        if m.status != "finished" or m.ft_home_goals is None:
            continue
        hg, ag = m.ft_home_goals, m.ft_away_goals
        stats[m.home_team_id]["p"] += 1
        stats[m.home_team_id]["gf"] += hg
        stats[m.home_team_id]["ga"] += ag
        stats[m.away_team_id]["p"] += 1
        stats[m.away_team_id]["gf"] += ag
        stats[m.away_team_id]["ga"] += hg
        if hg > ag:
            stats[m.home_team_id]["w"] += 1
            stats[m.away_team_id]["l"] += 1
        elif hg < ag:
            stats[m.away_team_id]["w"] += 1
            stats[m.home_team_id]["l"] += 1
        else:
            stats[m.home_team_id]["d"] += 1
            stats[m.away_team_id]["d"] += 1

    rows = []
    for team_id, s in stats.items():
        team = await db.get(Team, team_id)
        gd = s["gf"] - s["ga"]
        pts = 3 * s["w"] + s["d"]
        rows.append(StandingsRowOut(
            team=team.canonical_name if team else "Unknown",
            played=s["p"],
            won=s["w"],
            drawn=s["d"],
            lost=s["l"],
            goals_for=s["gf"],
            goals_against=s["ga"],
            goal_difference=gd,
            points=pts,
        ))

    rows.sort(key=lambda r: (-r.points, -r.goal_difference, -r.goals_for))
    return rows
