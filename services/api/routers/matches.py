"""Match detail endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import MarketPrediction, Match, Prediction, Team
from services.api.deps import get_db
from services.api.routers.leagues import _latest_prediction_summary
from services.api.schemas import (
    MarketOut,
    MatchDetailOut,
    RefereeStatsOut,
)
from services.engine.predict.utils import decompress_grid

router = APIRouter(prefix="/api/v1/matches", tags=["matches"])

# Market group mapping for accordion sections
MARKET_GROUPS = {
    "goals": [
        "over_under", "btts", "btts_over_25", "exact_total_goals",
        "team_totals_home", "team_totals_away",
    ],
    "result": ["match_result", "double_chance", "draw_no_bet", "winning_margin"],
    "handicaps": ["european_handicap", "asian_handicap"],
    "score": ["correct_score", "odd_even"],
    "defence": ["clean_sheet", "win_to_nil"],
}


@router.get("/{match_id}", response_model=MatchDetailOut)
async def get_match(match_id: int, db: AsyncSession = Depends(get_db)):
    """Get full match detail with prediction, markets, grid, and referee stats."""
    match = await db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    home = await db.get(Team, match.home_team_id)
    away = await db.get(Team, match.away_team_id)

    # Prediction summary
    pred_summary = await _latest_prediction_summary(db, match.id)

    # Latest prediction for markets and grid
    pred_result = await db.execute(
        select(Prediction)
        .where(Prediction.match_id == match.id)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    )
    pred = pred_result.scalar_one_or_none()

    # Markets grouped by category
    markets: dict[str, list[MarketOut]] = {}
    grid_list: list[list[float]] | None = None

    if pred:
        # Load market predictions
        mp_result = await db.execute(
            select(MarketPrediction)
            .where(MarketPrediction.prediction_id == pred.id)
            .order_by(MarketPrediction.market, MarketPrediction.line, MarketPrediction.selection)
        )
        market_preds = mp_result.scalars().all()

        # Group markets by category
        market_by_type: dict[str, list[MarketOut]] = {}
        for mp in market_preds:
            fair_odds = round(1.0 / mp.probability, 2) if mp.probability > 0.001 else None
            market_out = MarketOut(
                market=mp.market,
                selection=mp.selection,
                probability=mp.probability,
                fair_odds=fair_odds,
                line=mp.line,
            )
            market_by_type.setdefault(mp.market, []).append(market_out)

        for group_name, market_types in MARKET_GROUPS.items():
            group_markets: list[MarketOut] = []
            for mt in market_types:
                if mt in market_by_type:
                    group_markets.extend(market_by_type[mt])
            if group_markets:
                markets[group_name] = group_markets

        # Grid
        if pred.grid_compressed:
            grid_arr = decompress_grid(pred.grid_compressed)
            grid_list = grid_arr.tolist()

    # Referee stats
    referee_stats = None
    if match.referee_id is not None:
        ref_result = await db.execute(text("""
            SELECT r.name,
                   COUNT(*) as matches_officiated,
                   AVG(m.home_yellows + m.away_yellows + m.home_reds + m.away_reds)
                       as avg_cards,
                   AVG(m.home_yellows + m.away_yellows) as avg_yellows,
                   AVG(m.home_reds + m.away_reds) as avg_reds
            FROM matches m JOIN referees r ON m.referee_id = r.id
            WHERE m.referee_id = :ref_id
              AND m.status = 'finished'
              AND m.home_yellows IS NOT NULL
            GROUP BY r.name
        """), {"ref_id": match.referee_id})
        row = ref_result.first()
        if row and row.matches_officiated > 0:
            referee_stats = RefereeStatsOut(
                name=row.name,
                matches_officiated=row.matches_officiated,
                avg_cards_per_match=round(float(row.avg_cards), 2),
                avg_yellows_per_match=round(float(row.avg_yellows), 2),
                avg_reds_per_match=round(float(row.avg_reds), 2),
            )

    return MatchDetailOut(
        id=match.id,
        home_team=home.canonical_name if home else "Unknown",
        away_team=away.canonical_name if away else "Unknown",
        kickoff_utc=match.kickoff_utc,
        status=match.status,
        ft_home_goals=match.ft_home_goals,
        ft_away_goals=match.ft_away_goals,
        referee=referee_stats,
        prediction=pred_summary,
        markets=markets,
        grid=grid_list,
    )
