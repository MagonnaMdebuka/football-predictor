"""Walk-forward backtest engine.

Implements an expanding-window walk-forward strategy: for each distinct
kickoff date in the held-out seasons, fit the model on all matches
strictly before that date, then predict that date's matches.

Supports two refit schedules:
- per_date: refit on every distinct match date (most accurate, gate default)
- weekly: bucket by most recent Monday, refit only when bucket changes

Includes warm-start optimisation: when the team set is unchanged between
consecutive refits, the previous fit's parameter vector is passed as x0.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from services.engine.backtest.baselines import (
    base_rate_from_training,
)
from services.engine.backtest.gate import run_gate_checks
from services.engine.backtest.matchday import assign_matchdays
from services.engine.backtest.metrics import compute_metric_set
from services.engine.backtest.odds import extract_probabilities
from services.engine.backtest.report import SCHEMA_VERSION, get_git_commit
from services.engine.backtest.types import (
    BacktestConfig,
    BacktestReport,
    MatchPrediction,
    MetricSet,
    SeasonMetrics,
)
from services.engine.models.decay import time_weights
from services.engine.models.fit import fit_dixon_coles
from services.engine.models.grid import build_grid
from services.engine.models.params import pack


def _refit_date(match_date: pd.Timestamp, config: BacktestConfig) -> pd.Timestamp:
    """Compute the refit date for a given match date based on refit_step.

    For 'per_date': returns the match date itself.
    For 'weekly': returns the most recent weekday matching weekly_refit_day
    on or before the match date.
    """
    if config.refit_step == "per_date":
        return match_date

    # Weekly: find most recent Monday (or configured day) on or before match_date
    current_weekday = match_date.weekday()
    days_back = (current_weekday - config.weekly_refit_day) % 7
    return match_date - pd.Timedelta(days=days_back)


def _compute_metrics_from_predictions(
    predictions: list[MatchPrediction],
    source: str,
) -> MetricSet:
    """Compute MetricSet from a list of MatchPredictions for a given source.

    Args:
        predictions: list of match predictions
        source: 'model', 'uniform', 'base_rate', 'independent_poisson', or 'bookmaker'

    Returns:
        MetricSet for the given source.
    """
    if not predictions:
        return MetricSet(rps=0.0, log_loss=0.0, brier_home=0.0, brier_draw=0.0,
                         brier_away=0.0, n_matches=0, hit_rate=0.0)

    actual = np.array([p.result for p in predictions])

    if source == "bookmaker":
        # Filter to predictions with bookmaker odds
        valid = [(p, i) for i, p in enumerate(predictions) if p.bookmaker_home is not None]
        if not valid:
            return MetricSet(rps=0.0, log_loss=0.0, brier_home=0.0, brier_draw=0.0,
                             brier_away=0.0, n_matches=0, hit_rate=0.0)
        preds_valid, indices = zip(*valid)
        actual_valid = np.array([p.result for p in preds_valid])
        p_home = np.array([p.bookmaker_home for p in preds_valid], dtype=np.float64)
        p_draw = np.array([p.bookmaker_draw for p in preds_valid], dtype=np.float64)
        p_away = np.array([p.bookmaker_away for p in preds_valid], dtype=np.float64)
        vals = compute_metric_set(p_home, p_draw, p_away, actual_valid)
        return MetricSet(*vals)

    prefix = source + "_"
    if source == "independent_poisson":
        prefix = "indep_poisson_"

    p_home = np.array([getattr(p, prefix + "home") for p in predictions], dtype=np.float64)
    p_draw = np.array([getattr(p, prefix + "draw") for p in predictions], dtype=np.float64)
    p_away = np.array([getattr(p, prefix + "away") for p in predictions], dtype=np.float64)

    vals = compute_metric_set(p_home, p_draw, p_away, actual)
    return MetricSet(*vals)


def _build_season_metrics(
    season: str,
    predictions: list[MatchPrediction],
    early_predictions: list[MatchPrediction] | None = None,
) -> SeasonMetrics:
    """Build SeasonMetrics from predictions for a single season or combined."""
    model = _compute_metrics_from_predictions(predictions, "model")
    uniform = _compute_metrics_from_predictions(predictions, "uniform")
    base_rate = _compute_metrics_from_predictions(predictions, "base_rate")
    indep = _compute_metrics_from_predictions(predictions, "independent_poisson")
    bk = _compute_metrics_from_predictions(predictions, "bookmaker")

    bk_exclusion = sum(1 for p in predictions if p.bookmaker_home is None)
    bk_metrics = bk if bk.n_matches > 0 else None

    early = None
    if early_predictions:
        early = _compute_metrics_from_predictions(early_predictions, "model")

    return SeasonMetrics(
        season=season,
        model=model,
        uniform=uniform,
        base_rate=base_rate,
        independent_poisson=indep,
        bookmaker=bk_metrics,
        bookmaker_exclusion_count=bk_exclusion,
        early_season=early,
    )


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig,
    created_at: str | None = None,
    progress_callback: callable | None = None,
) -> BacktestReport:
    """Run a walk-forward backtest.

    For each distinct match date in the held-out seasons, fits the model
    on all historical data strictly before that date (with time decay),
    then predicts that date's matches.

    Args:
        df: full dataset with columns: date, season, home_team, away_team,
            home_goals, away_goals, ftr, source_row_raw (optional)
        config: backtest configuration
        created_at: ISO timestamp for the report (auto-generated if None)

    Returns:
        BacktestReport with all predictions, metrics, and gate results.
    """
    from datetime import datetime

    if created_at is None:
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    git_commit = get_git_commit()

    # Separate training-eligible and held-out data
    held_out_mask = df["season"].isin(config.held_out_seasons)
    held_out_df = df[held_out_mask].copy()
    all_data = df.copy()

    # Assign matchdays for held-out data
    held_out_df = held_out_df.copy()
    held_out_df["matchday"] = assign_matchdays(held_out_df, "season")

    # Get distinct prediction dates (sorted)
    prediction_dates = sorted(held_out_df["date"].dt.normalize().unique())

    # Determine refit dates and group prediction dates by refit date
    refit_groups: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for pred_date in prediction_dates:
        rd = _refit_date(pd.Timestamp(pred_date), config)
        refit_groups.setdefault(rd, []).append(pd.Timestamp(pred_date))

    # Walk forward
    all_predictions: list[MatchPrediction] = []
    prev_packed: NDArray[np.float64] | None = None
    prev_teams: list[str] | None = None

    sorted_refit_dates = sorted(refit_groups.keys())
    n_refits = len(sorted_refit_dates)

    for refit_idx, refit_date in enumerate(sorted_refit_dates):
        pred_dates = refit_groups[refit_date]

        if progress_callback is not None:
            progress_callback(refit_idx + 1, n_refits, refit_date)

        # Training data: everything strictly before the earliest prediction date in this group
        cutoff = min(pred_dates)
        training_mask = all_data["date"].dt.normalize() < cutoff
        training_df = all_data[training_mask].copy()

        if len(training_df) < 10:
            continue

        # Time decay weights
        match_dates = training_df["date"].values.astype("datetime64[D]")
        ref_date = pd.Timestamp(cutoff).to_pydatetime().date()
        weights = time_weights(match_dates, ref_date, config.xi)

        # Warm-start: use previous params if team set unchanged
        x0 = None
        teams = sorted(
            set(training_df["home_team"].unique()) | set(training_df["away_team"].unique())
        )
        if prev_packed is not None and prev_teams is not None and teams == prev_teams:
            x0 = prev_packed

        # Fit main model
        fit_result = fit_dixon_coles(
            training_df, weights=weights, rho_bounds=config.rho_bounds, x0=x0,
        )
        prev_packed = pack(fit_result.params)
        prev_teams = teams

        # Fit independent Poisson (rho=0)
        indep_result = fit_dixon_coles(
            training_df, weights=weights, rho_bounds=(0.0, 0.0),
        )

        # Base rate from training
        br = base_rate_from_training(training_df)

        # Generate predictions for each prediction date in this group
        for pred_date in pred_dates:
            day_mask = held_out_df["date"].dt.normalize() == pred_date
            day_matches = held_out_df[day_mask]

            for _, match in day_matches.iterrows():
                ht = match["home_team"]
                at = match["away_team"]

                # Skip matches involving teams not in the training data
                # (promoted teams on their first matchday)
                fitted_teams = fit_result.params.teams
                if ht not in fitted_teams or at not in fitted_teams:
                    continue

                # Model probabilities
                grid = build_grid(fit_result.params, ht, at)
                model_h = grid.home_win
                model_d = grid.draw
                model_a = grid.away_win

                # Independent Poisson probabilities
                indep_teams = indep_result.params.teams
                if ht not in indep_teams or at not in indep_teams:
                    continue
                indep_grid = build_grid(indep_result.params, ht, at)
                indep_h = indep_grid.home_win
                indep_d = indep_grid.draw
                indep_a = indep_grid.away_win

                # Bookmaker probabilities
                bk_h, bk_d, bk_a = None, None, None
                if "source_row_raw" in match.index and match["source_row_raw"] is not None:
                    bk_probs = extract_probabilities(
                        match["source_row_raw"], config.bookmaker_odds_cols,
                    )
                    if bk_probs is not None:
                        bk_h, bk_d, bk_a = bk_probs

                pred = MatchPrediction(
                    date=pd.Timestamp(match["date"]).strftime("%Y-%m-%d"),
                    season=str(match["season"]),
                    home_team=ht,
                    away_team=at,
                    home_goals=int(match["home_goals"]),
                    away_goals=int(match["away_goals"]),
                    result=str(match["ftr"]),
                    matchday=int(match["matchday"]),
                    model_home=model_h,
                    model_draw=model_d,
                    model_away=model_a,
                    uniform_home=1.0 / 3.0,
                    uniform_draw=1.0 / 3.0,
                    uniform_away=1.0 / 3.0,
                    base_rate_home=br.home,
                    base_rate_draw=br.draw,
                    base_rate_away=br.away,
                    indep_poisson_home=indep_h,
                    indep_poisson_draw=indep_d,
                    indep_poisson_away=indep_a,
                    bookmaker_home=bk_h,
                    bookmaker_draw=bk_d,
                    bookmaker_away=bk_a,
                    n_training_matches=len(training_df),
                )
                all_predictions.append(pred)

    # Build per-season metrics
    season_predictions: dict[str, list[MatchPrediction]] = {}
    for p in all_predictions:
        season_predictions.setdefault(p.season, []).append(p)

    season_metrics = []
    for season in config.held_out_seasons:
        preds = season_predictions.get(season, [])
        early_preds = [p for p in preds if p.matchday <= 6]
        sm = _build_season_metrics(season, preds, early_preds if early_preds else None)
        season_metrics.append(sm)

    # Combined metrics
    all_early = [p for p in all_predictions if p.matchday <= 6]
    combined = _build_season_metrics("combined", all_predictions,
                                     all_early if all_early else None)

    # Early-season combined
    early_combined = None
    if all_early:
        early_combined = _build_season_metrics("early_season_combined", all_early)

    # Gate checks
    gate_passed, gate_details = run_gate_checks(combined)

    return BacktestReport(
        schema_version=SCHEMA_VERSION,
        created_at=created_at,
        git_commit=git_commit,
        config=config,
        seasons=season_metrics,
        combined=combined,
        early_season_combined=early_combined,
        predictions=all_predictions,
        gate_passed=gate_passed,
        gate_details=gate_details,
    )
