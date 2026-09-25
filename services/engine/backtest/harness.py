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
from services.engine.backtest.count_gate import run_count_gate_checks
from services.engine.backtest.count_harness import (
    run_compound_card_predictions,
    run_count_predictions,
)
from services.engine.backtest.count_types import (
    CountPrediction,
    compute_count_brier,
    compute_count_calibration,
)
from services.engine.backtest.gate import run_gate_checks
from services.engine.backtest.matchday import assign_matchdays
from services.engine.backtest.metrics import compute_metric_set
from services.engine.backtest.odds import extract_probabilities
from services.engine.backtest.report import SCHEMA_VERSION, get_git_commit
from services.engine.backtest.types import (
    BacktestConfig,
    BacktestReport,
    CountCalibrationSummary,
    CountMetricSummary,
    GoalCalibration,
    MatchPrediction,
    MetricSet,
    SeasonMetrics,
)
from services.engine.config.league_defaults import get_league_config
from services.engine.models.decay import time_weights
from services.engine.models.fit import fit_dixon_coles
from services.engine.models.grid import build_grid
from services.engine.models.params import DixonColesParams, pack


def _augment_params_with_fallback(
    params: DixonColesParams,
    teams_needed: list[str],
) -> tuple[DixonColesParams, list[str]]:
    """Return params with missing teams added at league-average strength.

    Teams not in params.teams are appended with attack=0.0, defence=0.0
    (league average). Returns the augmented params and list of fallback teams.
    """
    missing = [t for t in teams_needed if t not in params.teams]
    if not missing:
        return params, []

    new_teams = list(params.teams) + missing
    new_attack = np.concatenate([params.attack, np.zeros(len(missing))])
    new_defence = np.concatenate([params.defence, np.zeros(len(missing))])

    return DixonColesParams(
        teams=new_teams,
        mu=params.mu,
        attack=new_attack,
        defence=new_defence,
        gamma=params.gamma,
        rho=params.rho,
    ), missing


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
    ablation = _compute_metrics_from_predictions(predictions, "ablation")
    bk = _compute_metrics_from_predictions(predictions, "bookmaker")

    bk_exclusion = sum(1 for p in predictions if p.bookmaker_home is None)
    bk_metrics = bk if bk.n_matches > 0 else None
    ablation_metrics = ablation if ablation.n_matches > 0 else None

    early = None
    if early_predictions:
        early = _compute_metrics_from_predictions(early_predictions, "model")

    return SeasonMetrics(
        season=season,
        model=model,
        uniform=uniform,
        base_rate=base_rate,
        independent_poisson=indep,
        ablation=ablation_metrics,
        bookmaker=bk_metrics,
        bookmaker_exclusion_count=bk_exclusion,
        early_season=early,
    )


def _compute_goal_calibration(
    predictions: list[MatchPrediction],
) -> GoalCalibration | None:
    """Compute aggregate goal-calibration metrics from predictions.

    Compares model-predicted goal totals (lambda_home + lambda_away) against
    actual totals, and predicted P(over 2.5) against the observed over-2.5 rate.
    """
    if not predictions:
        return None

    n = len(predictions)

    # Predicted mean total goals (from lambdas)
    pred_totals = [p.lambda_home + p.lambda_away for p in predictions]
    predicted_mean = sum(pred_totals) / n

    # Actual mean total goals
    actual_totals = [p.home_goals + p.away_goals for p in predictions]
    actual_mean = sum(actual_totals) / n

    # Predicted P(over 2.5): sum cells where i+j > 2 in each match's grid.
    # We approximate from the independent Poisson using the lambdas, since
    # the grid is not stored. P(total <= 2) = sum of Poisson PMFs for 0,1,2.
    from services.engine.models.poisson import poisson_pmf

    pred_o25_rates = []
    for p in predictions:
        p_under = 0.0
        for total in range(3):  # 0, 1, 2
            for h in range(total + 1):
                a = total - h
                p_under += float(poisson_pmf(h, p.lambda_home)) * float(
                    poisson_pmf(a, p.lambda_away)
                )
        pred_o25_rates.append(1.0 - p_under)
    predicted_o25 = sum(pred_o25_rates) / n

    # Actual over-2.5 rate
    actual_o25 = sum(1 for t in actual_totals if t > 2) / n

    return GoalCalibration(
        n_matches=n,
        predicted_mean_total=round(predicted_mean, 4),
        actual_mean_total=round(actual_mean, 4),
        predicted_over_25_rate=round(predicted_o25, 4),
        actual_over_25_rate=round(actual_o25, 4),
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

        # Fit independent Poisson (rho=0, no time decay)
        indep_result = fit_dixon_coles(
            training_df, weights=None, rho_bounds=(0.0, 0.0),
        )

        # Fit ablation: no time decay, rho fitted (isolates tau contribution)
        ablation_result = fit_dixon_coles(
            training_df, weights=None, rho_bounds=config.rho_bounds,
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

                # Augment params for teams missing from training data
                # (promoted teams use league-average attack=0, defence=0)
                model_params, fallback = _augment_params_with_fallback(
                    fit_result.params, [ht, at],
                )
                indep_params, _ = _augment_params_with_fallback(
                    indep_result.params, [ht, at],
                )
                ablation_params, _ = _augment_params_with_fallback(
                    ablation_result.params, [ht, at],
                )

                # Model probabilities
                grid = build_grid(model_params, ht, at)
                model_h = grid.home_win
                model_d = grid.draw
                model_a = grid.away_win

                # Independent Poisson probabilities
                indep_grid = build_grid(indep_params, ht, at)
                indep_h = indep_grid.home_win
                indep_d = indep_grid.draw
                indep_a = indep_grid.away_win

                # Ablation probabilities (no decay, rho fitted)
                ablation_grid = build_grid(ablation_params, ht, at)
                abl_h = ablation_grid.home_win
                abl_d = ablation_grid.draw
                abl_a = ablation_grid.away_win

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
                    ablation_home=abl_h,
                    ablation_draw=abl_d,
                    ablation_away=abl_a,
                    lambda_home=grid.lambda_home,
                    lambda_away=grid.lambda_away,
                    n_training_matches=len(training_df),
                    fallback_teams=fallback,
                )
                all_predictions.append(pred)

    assert len(all_predictions) == len(held_out_df), (
        f"Expected {len(held_out_df)} predictions, got {len(all_predictions)}"
    )

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

    # Goal calibration
    goal_cal = _compute_goal_calibration(all_predictions)

    # Gate checks
    gate_passed, gate_details = run_gate_checks(combined)

    baseline_configs = {
        "model": {"xi": config.xi, "rho": "fitted"},
        "independent_poisson": {"xi": 0, "rho": 0},
        "ablation": {"xi": 0, "rho": "fitted"},
        "base_rate": {"description": "training proportions"},
        "uniform": {"description": "1/3 each"},
        "bookmaker": {"description": "market closing odds"},
    }

    # Count models (corners and cards) — conditional on capability flags
    corner_preds: list[CountPrediction] = []
    card_preds: list[CountPrediction] = []
    corner_metrics_summary: CountMetricSummary | None = None
    card_metrics_summary: CountMetricSummary | None = None
    corner_cal_summary: CountCalibrationSummary | None = None
    card_cal_summary: CountCalibrationSummary | None = None
    corner_gate_passed: bool | None = None
    card_gate_passed: bool | None = None
    corner_gate_dets: list = []
    card_gate_dets: list = []

    has_corners = False
    has_cards = False
    if config.league_code:
        try:
            league_cfg = get_league_config(config.league_code)
            has_corners = league_cfg.has_corners
            has_cards = league_cfg.has_cards
        except KeyError:
            pass

    if has_corners or has_cards:
        # Run count models using the same walk-forward dates
        corner_prev_packed = None
        corner_prev_teams = None
        corner_prev_total_packed = None
        corner_prev_total_teams = None
        card_prev_yellow_packed = None
        card_prev_yellow_teams = None
        card_prev_red_packed = None
        card_prev_red_teams = None

        for refit_date in sorted_refit_dates:
            pred_dates = refit_groups[refit_date]
            cutoff = min(pred_dates)
            training_mask = all_data["date"].dt.normalize() < cutoff
            training_df = all_data[training_mask].copy()

            if len(training_df) < 10:
                continue

            # Time decay weights for count models
            match_dates_arr = training_df["date"].values.astype("datetime64[D]")
            ref_date_val = pd.Timestamp(cutoff).to_pydatetime().date()
            corners_xi = config.corners_xi if config.corners_xi is not None else config.xi
            cards_xi = config.cards_xi if config.cards_xi is not None else config.xi

            # Gather all prediction matches for this refit date
            day_matches_list = []
            for pred_date in pred_dates:
                day_mask = held_out_df["date"].dt.normalize() == pred_date
                day_matches_list.append(held_out_df[day_mask])
            if not day_matches_list:
                continue
            pred_matches = pd.concat(day_matches_list)

            if has_corners:
                corner_weights = time_weights(match_dates_arr, ref_date_val, corners_xi)
                c_preds, c_packed, c_teams, c_t_packed, c_t_teams = run_count_predictions(
                    training_df, pred_matches, model_type="corners",
                    weights=corner_weights,
                    prev_packed=corner_prev_packed,
                    prev_teams=corner_prev_teams,
                    prev_total_packed=corner_prev_total_packed,
                    prev_total_teams=corner_prev_total_teams,
                )
                corner_preds.extend(c_preds)
                if len(c_packed) > 0:
                    corner_prev_packed = c_packed
                    corner_prev_teams = c_teams
                if c_t_packed is not None:
                    corner_prev_total_packed = c_t_packed
                    corner_prev_total_teams = c_t_teams

            if has_cards:
                card_weights = time_weights(match_dates_arr, ref_date_val, cards_xi)
                k_preds, yp, yt, rp, rt = run_compound_card_predictions(
                    training_df, pred_matches,
                    weights=card_weights,
                    min_referee_matches=config.min_referee_matches,
                    prev_yellow_packed=card_prev_yellow_packed,
                    prev_yellow_teams=card_prev_yellow_teams,
                    prev_red_packed=card_prev_red_packed,
                    prev_red_teams=card_prev_red_teams,
                )
                card_preds.extend(k_preds)
                if len(yp) > 0:
                    card_prev_yellow_packed = yp
                    card_prev_yellow_teams = yt
                if len(rp) > 0:
                    card_prev_red_packed = rp
                    card_prev_red_teams = rt

        # Compute count metrics and gates
        if corner_preds:
            corner_ms = compute_count_brier(corner_preds)
            corner_metrics_summary = CountMetricSummary(
                mean_brier=corner_ms.mean_brier,
                per_line_brier=corner_ms.per_line_brier,
                n_predictions=corner_ms.n_predictions,
                mean_predicted_total=corner_ms.mean_predicted_total,
                mean_actual_total=corner_ms.mean_actual_total,
            )
            corner_cal = compute_count_calibration(corner_preds)
            if corner_cal:
                corner_cal_summary = CountCalibrationSummary(
                    n_predictions=corner_cal.n_predictions,
                    predicted_mean_total=corner_cal.predicted_mean_total,
                    actual_mean_total=corner_cal.actual_mean_total,
                    bias=corner_cal.bias,
                )
            corner_gate_passed, corner_gate_dets = run_count_gate_checks(
                corner_preds, corner_ms, "corners",
            )

        if card_preds:
            card_ms = compute_count_brier(card_preds)
            card_metrics_summary = CountMetricSummary(
                mean_brier=card_ms.mean_brier,
                per_line_brier=card_ms.per_line_brier,
                n_predictions=card_ms.n_predictions,
                mean_predicted_total=card_ms.mean_predicted_total,
                mean_actual_total=card_ms.mean_actual_total,
            )
            card_cal = compute_count_calibration(card_preds)
            if card_cal:
                card_cal_summary = CountCalibrationSummary(
                    n_predictions=card_cal.n_predictions,
                    predicted_mean_total=card_cal.predicted_mean_total,
                    actual_mean_total=card_cal.actual_mean_total,
                    bias=card_cal.bias,
                )
            card_gate_passed, card_gate_dets = run_count_gate_checks(
                card_preds, card_ms, "cards",
            )

    # Convert count predictions to serialisable dicts
    from dataclasses import asdict
    corner_pred_dicts = [asdict(p) for p in corner_preds]
    card_pred_dicts = [asdict(p) for p in card_preds]

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
        baseline_configs=baseline_configs,
        goal_calibration=goal_cal,
        corner_predictions=corner_pred_dicts,
        card_predictions=card_pred_dicts,
        corner_metrics=corner_metrics_summary,
        card_metrics=card_metrics_summary,
        corner_calibration=corner_cal_summary,
        card_calibration=card_cal_summary,
        corner_gate_passed=corner_gate_passed,
        card_gate_passed=card_gate_passed,
        corner_gate_details=corner_gate_dets,
        card_gate_details=card_gate_dets,
    )
