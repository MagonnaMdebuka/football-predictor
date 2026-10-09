"""Predict command orchestration.

Queries DB for training data and scheduled fixtures, fits the Dixon-Coles
model, and writes predictions + market predictions back to the database.
Engine functions remain pure — all DB access is in this module.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db.models import (
    League,
    MarketPrediction,
    Match,
    ModelRun,
    Prediction,
    Team,
    TeamStrength,
)
from services.engine.backtest.harness import _augment_params_with_fallback
from services.engine.backtest.report import get_git_commit
from services.engine.config.league_defaults import get_league_config
from services.engine.markets.goals import grid_to_markets
from services.engine.models.decay import time_weights
from services.engine.models.fit import fit_dixon_coles
from services.engine.models.grid import build_grid
from services.engine.predict.entropy import compute_confidence
from services.engine.predict.scoreline import check_scoreline_disagreement
from services.engine.predict.utils import (
    compress_grid,
    compute_fingerprint,
    decompress_grid,
    flatten_markets,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "dixon_coles"
MODEL_VERSION = "1.0.0"


# ── Training data query ───────────────────────────────────────────────


def build_training_df(session: Session, league_id: int) -> pd.DataFrame:
    """Query finished matches for a league and return an engine-ready DataFrame."""
    sql = text("""
        SELECT m.kickoff_utc AS date,
               ht.canonical_name AS home_team,
               at.canonical_name AS away_team,
               m.ft_home_goals AS home_goals,
               m.ft_away_goals AS away_goals
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.league_id = :league_id
          AND m.status = 'finished'
          AND m.ft_home_goals IS NOT NULL
        ORDER BY m.kickoff_utc
    """)
    result = session.execute(sql, {"league_id": league_id})
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team",
                                     "home_goals", "away_goals"])
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


# ── Main orchestration ────────────────────────────────────────────────


def run_predict(
    league_code: str = "E0",
    days_ahead: int = 14,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Fit model on finished matches, predict upcoming fixtures, write to DB."""
    from services.engine.ingest.db_session import get_session

    with get_session() as session:
        # 1. Look up league
        league = session.execute(
            select(League).where(League.fd_couk_code == league_code)
        ).scalar_one_or_none()
        if league is None:
            logger.error("League %s not found in database", league_code)
            return

        # 2. Build training data
        df = build_training_df(session, league.id)
        n_matches = len(df)
        if n_matches < 10:
            logger.error("Only %d finished matches — need at least 10", n_matches)
            return

        latest_date = df["date"].max().date()
        logger.info(
            "Training data: %d matches, latest %s", n_matches, latest_date,
        )

        # 3. Check fingerprint for idempotency
        git_commit = get_git_commit()
        fingerprint = compute_fingerprint(
            league.id, n_matches, latest_date,
            model_version=MODEL_VERSION,
            git_commit=git_commit or "",
        )
        if not force:
            existing = session.execute(
                select(ModelRun).where(ModelRun.fingerprint == fingerprint)
            ).scalar_one_or_none()
            if existing is not None:
                logger.info("No new data since last run (fingerprint %s…)", fingerprint[:12])
                return
        else:
            logger.info("Force flag set — bypassing fingerprint check")

        # 4. Load league config and fit model
        config = get_league_config(league_code)
        ref_date = datetime.now(timezone.utc).date()
        match_dates = df["date"].values.astype("datetime64[D]")
        weights = time_weights(match_dates, ref_date, config.xi)

        logger.info("Fitting Dixon-Coles model (xi=%.4f)...", config.xi)
        fit_result = fit_dixon_coles(df, weights=weights, rho_bounds=config.rho_bounds)
        params = fit_result.params

        if dry_run:
            logger.info("[DRY RUN] Model fit complete, skipping DB writes")
            return

        # 5. Write ModelRun
        # When --force is used the fingerprint may already exist; replace the
        # last 15 chars with a timestamp suffix to stay within varchar(64).
        if force:
            existing = session.execute(
                select(ModelRun).where(ModelRun.fingerprint == fingerprint)
            ).scalar_one_or_none()
            if existing is not None:
                ts = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
                fingerprint = fingerprint[:50] + f"_f{ts}"
        model_run = ModelRun(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            league_id=league.id,
            fingerprint=fingerprint,
            n_training_matches=n_matches,
            git_commit=git_commit,
            parameters=json.dumps({
                "mu": float(params.mu),
                "gamma": float(params.gamma),
                "rho": float(params.rho),
                "xi": config.xi,
                "n_teams": len(params.teams),
            }),
            training_window_start=df["date"].min().to_pydatetime(),
            training_window_end=df["date"].max().to_pydatetime(),
        )
        session.add(model_run)
        session.flush()  # get model_run.id

        # 6. Write TeamStrength rows
        team_name_to_id: dict[str, int] = {}
        teams_in_db = session.execute(select(Team)).scalars().all()
        for t in teams_in_db:
            team_name_to_id[t.canonical_name] = t.id

        for i, team_name in enumerate(params.teams):
            team_id = team_name_to_id.get(team_name)
            if team_id is None:
                continue
            session.add(TeamStrength(
                model_run_id=model_run.id,
                team_id=team_id,
                attack=float(params.attack[i]),
                defence=float(params.defence[i]),
                home_advantage=float(params.gamma),
            ))

        # 7. Query scheduled fixtures
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(days=days_ahead)
        fixtures = session.execute(
            select(Match)
            .where(Match.league_id == league.id)
            .where(Match.status == "scheduled")
            .where(Match.kickoff_utc >= now)
            .where(Match.kickoff_utc <= window_end)
            .order_by(Match.kickoff_utc)
        ).scalars().all()

        if not fixtures:
            logger.info("No scheduled fixtures in the next %d days", days_ahead)
            session.commit()
            return

        logger.info("Generating predictions for %d fixtures...", len(fixtures))

        # 8. Predict each fixture
        n_predictions = 0
        for match in fixtures:
            home_team = session.get(Team, match.home_team_id)
            away_team = session.get(Team, match.away_team_id)
            if home_team is None or away_team is None:
                logger.warning("Skipping match %d — missing team", match.id)
                continue

            ht = home_team.canonical_name
            at = away_team.canonical_name

            augmented, fallback = _augment_params_with_fallback(params, [ht, at])
            if fallback:
                logger.info("Fallback (league avg) for: %s", ", ".join(fallback))

            grid = build_grid(augmented, ht, at)
            markets = grid_to_markets(grid)
            confidence = compute_confidence(grid.home_win, grid.draw, grid.away_win)
            disagreement = check_scoreline_disagreement(grid)

            modal_h, modal_a, modal_prob = grid.most_likely_score()

            prediction = Prediction(
                model_run_id=model_run.id,
                match_id=match.id,
                home_win_prob=grid.home_win,
                draw_prob=grid.draw,
                away_win_prob=grid.away_win,
                home_expected_goals=grid.lambda_home,
                away_expected_goals=grid.lambda_away,
                grid_compressed=compress_grid(grid.grid),
                n_matches_train=n_matches,
            )
            session.add(prediction)
            session.flush()  # get prediction.id

            market_rows = flatten_markets(markets, prediction.id)
            for row in market_rows:
                session.add(MarketPrediction(**row))

            n_predictions += 1
            logger.info(
                "  %s vs %s: H=%.1f%% D=%.1f%% A=%.1f%% (%s) "
                "score=%d-%d (%.0f%%) confidence=%s",
                ht, at,
                grid.home_win * 100, grid.draw * 100, grid.away_win * 100,
                "DISAGREE" if disagreement else "OK",
                modal_h, modal_a, modal_prob * 100,
                confidence,
            )

        session.commit()
        logger.info(
            "Done: %d predictions written (model_run=%d, fingerprint=%s…)",
            n_predictions, model_run.id, fingerprint[:12],
        )
