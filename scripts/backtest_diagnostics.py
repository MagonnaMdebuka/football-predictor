"""Phase 5 diagnostic backtest: corners, cards, referee effects, capability flags.

Runs the full walk-forward backtest with count models enabled, then reports:
1. Fitted alpha vs raw variance-to-mean ratio
2. Calibration: predicted vs actual totals
3. Brier scores for specific O/U lines vs baseline
4. Top/bottom 5 referees by fitted card effect
5. Capability flag verification
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.backtest.count_gate import _compute_baseline_brier
from services.engine.backtest.count_types import compute_count_brier
from services.engine.backtest.harness import run_backtest
from services.engine.backtest.types import BacktestConfig
from services.engine.config import get_league_config
from services.engine.config.league_defaults import LeagueConfig, LEAGUE_CONFIGS
from services.engine.models.count_fit import fit_count_model
from services.engine.models.decay import time_weights


def load_all_seasons() -> pd.DataFrame:
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
    )
    season_files = {
        "2019-20": "E0_1920.csv",
        "2020-21": "E0_2021.csv",
        "2021-22": "E0_2122.csv",
        "2022-23": "E0_2223.csv",
        "2023-24": "E0_2324.csv",
        "2024-25": "E0_2425.csv",
        "2025-26": "E0_2526.csv",
    }

    frames = []
    for season, fname in season_files.items():
        path = os.path.join(data_dir, fname)
        raw = pd.read_csv(path, encoding="utf-8-sig")
        raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

        odds_cols = ["PSCH", "PSCD", "PSCA", "AvgCH", "AvgCD", "AvgCA"]
        avail = [c for c in odds_cols if c in raw.columns]

        df = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], dayfirst=True),
            "season": season,
            "home_team": raw["HomeTeam"],
            "away_team": raw["AwayTeam"],
            "home_goals": raw["FTHG"].astype(int),
            "away_goals": raw["FTAG"].astype(int),
            "ftr": raw["FTR"],
        })
        if avail:
            df["source_row_raw"] = raw[avail].apply(
                lambda row: json.dumps(
                    {k: v for k, v in row.items() if pd.notna(v)}
                ),
                axis=1,
            )

        # Corner columns
        if "HC" in raw.columns and "AC" in raw.columns:
            df["home_corners"] = pd.to_numeric(raw["HC"], errors="coerce").astype("Int64")
            df["away_corners"] = pd.to_numeric(raw["AC"], errors="coerce").astype("Int64")

        # Card columns: individual yellow/red counts and derived booking points
        card_cols = {"HY", "AY", "HR", "AR"}
        if card_cols.issubset(raw.columns):
            hy = pd.to_numeric(raw["HY"], errors="coerce").fillna(0).astype(int)
            ay = pd.to_numeric(raw["AY"], errors="coerce").fillna(0).astype(int)
            hr = pd.to_numeric(raw["HR"], errors="coerce").fillna(0).astype(int)
            ar = pd.to_numeric(raw["AR"], errors="coerce").fillna(0).astype(int)
            df["home_yellows"] = hy
            df["away_yellows"] = ay
            df["home_reds"] = hr
            df["away_reds"] = ar
            df["home_booking_points"] = 10 * hy + 25 * hr
            df["away_booking_points"] = 10 * ay + 25 * ar

        # Referee column
        if "Referee" in raw.columns:
            df["referee"] = raw["Referee"].astype(str).str.strip()

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def progress(step: int, total: int, refit_date) -> None:
    print(f"  Refit {step}/{total}  cutoff={refit_date.date()}", flush=True)


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    df = load_all_seasons()
    print(f"Loaded {len(df)} matches across {df['season'].nunique()} seasons", flush=True)

    # ================================================================
    # PART 0: Raw data variance-to-mean ratios
    # ================================================================
    section("PART 0: Raw Data Variance-to-Mean Ratios")

    # Use all seasons for the overview
    all_corners = pd.concat([df["home_corners"], df["away_corners"]]).dropna()
    corner_mean = float(all_corners.mean())
    corner_var = float(all_corners.var())
    corner_vmr = corner_var / corner_mean

    all_bp = pd.concat([df["home_booking_points"], df["away_booking_points"]]).dropna()
    bp_mean = float(all_bp.mean())
    bp_var = float(all_bp.var())
    bp_vmr = bp_var / bp_mean

    print(f"\nCorners (per team, all seasons):")
    print(f"  Mean:  {corner_mean:.3f}")
    print(f"  Var:   {corner_var:.3f}")
    print(f"  VMR:   {corner_vmr:.3f}  (>1 = overdispersed vs Poisson)")

    print(f"\nBooking points (per team, all seasons):")
    print(f"  Mean:  {bp_mean:.3f}")
    print(f"  Var:   {bp_var:.3f}")
    print(f"  VMR:   {bp_vmr:.3f}  (>1 = overdispersed vs Poisson)")

    # Also show raw yellow/red counts
    all_y = pd.concat([
        pd.to_numeric(df.get("home_booking_points", pd.Series(dtype=float))),
    ]).dropna()

    # ================================================================
    # PART 0b: Fit a single snapshot model to inspect alpha
    # ================================================================
    section("PART 0b: Snapshot Fit — Alpha Recovery")

    # Fit on training data (pre-2024-25)
    train_mask = ~df["season"].isin(["2024-25", "2025-26"])
    train_df = df[train_mask].dropna(subset=["home_corners", "away_corners"])
    print(f"\nCorner model training set: {len(train_df)} matches")

    corner_fit = fit_count_model(
        train_df,
        target_home_col="home_corners",
        target_away_col="away_corners",
    )
    print(f"  alpha (corners): {corner_fit.params.alpha:.6f}")
    print(f"  converged: {corner_fit.converged}")
    print(f"  gamma (home adv): {corner_fit.params.gamma:.4f}")
    print(f"  mu (intercept): {corner_fit.params.mu:.4f}")
    if corner_fit.params.alpha < 1e-10:
        print("  WARNING: alpha collapsed to Poisson fallback!")
    else:
        print(f"  NB2 variance at mean={corner_mean:.1f}: "
              f"{corner_mean + corner_fit.params.alpha * corner_mean**2:.3f} "
              f"(vs Poisson: {corner_mean:.3f})")

    # Corner shrinkage diagnostic
    atk_std = float(corner_fit.params.attack.std())
    def_std = float(corner_fit.params.defence.std())
    atk_range = float(np.ptp(corner_fit.params.attack))
    def_range = float(np.ptp(corner_fit.params.defence))
    print(f"\n  Corner team-effect spread:")
    print(f"    attack.std():  {atk_std:.4f}  range: {atk_range:.4f}")
    print(f"    defence.std(): {def_std:.4f}  range: {def_range:.4f}")
    if atk_std < 0.05 and def_std < 0.05:
        print("    NOTE: Team effects are in the noise (std < 0.05) — "
              "model may be no better than a flat baseline for corners.")

    train_bp = df[train_mask].dropna(subset=["home_booking_points", "away_booking_points"])
    print(f"\nCard model training set: {len(train_bp)} matches")

    card_fit = fit_count_model(
        train_bp,
        target_home_col="home_booking_points",
        target_away_col="away_booking_points",
        include_referees=True,
        min_referee_matches=20,
    )
    print(f"  alpha (cards): {card_fit.params.alpha:.6f}")
    print(f"  converged: {card_fit.converged}")
    print(f"  gamma (home adv): {card_fit.params.gamma:.4f}")
    print(f"  mu (intercept): {card_fit.params.mu:.4f}")
    if card_fit.params.alpha < 1e-10:
        print("  WARNING: alpha collapsed to Poisson fallback!")
    else:
        print(f"  NB2 variance at mean={bp_mean:.1f}: "
              f"{bp_mean + card_fit.params.alpha * bp_mean**2:.3f} "
              f"(vs Poisson: {bp_mean:.3f})")

    # ================================================================
    # PART 4 (early): Referee effects from snapshot fit
    # ================================================================
    section("PART 4: Referee Effects (Snapshot Fit)")
    if card_fit.params.referees:
        ref_data = []
        ref_counts = train_bp["referee"].value_counts()
        for i, ref_name in enumerate(card_fit.params.referees):
            eff = card_fit.params.referee_effect[i]
            n_matches = int(ref_counts.get(ref_name, 0))
            ref_data.append((ref_name, eff, n_matches))

        ref_data.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{len(ref_data)} referees with 20+ matches fitted")
        print(f"\nTop 5 (strictest — positive effect = more cards):")
        for name, eff, n in ref_data[:5]:
            mult = np.exp(eff)
            print(f"  {name:25s}  effect={eff:+.4f}  multiplier={mult:.3f}x  (n={n})")

        print(f"\nBottom 5 (most lenient — negative effect = fewer cards):")
        for name, eff, n in ref_data[-5:]:
            mult = np.exp(eff)
            print(f"  {name:25s}  effect={eff:+.4f}  multiplier={mult:.3f}x  (n={n})")
    else:
        print("  No referees met the 20-match threshold.")

    # ================================================================
    # Run the full backtest with count models
    # ================================================================
    section("RUNNING FULL WALK-FORWARD BACKTEST")

    league = get_league_config("E0")
    config = BacktestConfig(
        held_out_seasons=league.test_seasons,
        training_start_season=league.training_start_season,
        xi=league.xi,
        rho_bounds=league.rho_bounds,
        max_goals=league.max_goals,
        refit_step="per_date",
        seed=42,
        league_code="E0",
    )

    print(f"\nConfig: xi={config.xi}, league={config.league_code}")
    print(f"Held-out: {config.held_out_seasons}")
    report = run_backtest(df, config, progress_callback=progress)
    elapsed = time.time() - t0
    print(f"\nBacktest completed in {elapsed:.0f}s")
    print(f"Goals predictions: {len(report.predictions)}")
    print(f"Corner predictions: {len(report.corner_predictions)}")
    print(f"Card predictions: {len(report.card_predictions)}")

    # ================================================================
    # PART 1: Fitted alpha from walk-forward
    # ================================================================
    section("PART 1: Fitted Dispersion (Alpha) from Walk-Forward Predictions")

    if report.corner_predictions:
        corner_alphas = [p["alpha"] for p in report.corner_predictions]
        print(f"\nCorners:")
        print(f"  Alpha range across predictions: [{min(corner_alphas):.6f}, {max(corner_alphas):.6f}]")
        print(f"  Alpha mean: {np.mean(corner_alphas):.6f}")
        print(f"  Raw VMR: {corner_vmr:.3f}")
        if min(corner_alphas) < 1e-10:
            print("  WARNING: Some predictions used Poisson fallback (alpha ~ 0)")
        else:
            print("  Corners are overdispersed (alpha > 0, VMR > 1). NB2 is appropriate.")

    if report.card_predictions:
        card_alphas = [p["alpha"] for p in report.card_predictions]
        print(f"\nCards/Booking Points:")
        print(f"  Alpha range across predictions: [{min(card_alphas):.6f}, {max(card_alphas):.6f}]")
        print(f"  Alpha mean: {np.mean(card_alphas):.6f}")
        print(f"  Raw VMR: {bp_vmr:.3f}")
        if min(card_alphas) < 1e-10:
            print("  WARNING: Some predictions used Poisson fallback (alpha ~ 0)")
        else:
            print("  Booking points are overdispersed (alpha > 0, VMR > 1). NB2 is appropriate.")

    # ================================================================
    # PART 2: Calibration
    # ================================================================
    section("PART 2: Calibration — Predicted vs Actual")

    if report.corner_calibration:
        cc = report.corner_calibration
        print(f"\nCorners (n={cc.n_predictions}):")
        print(f"  Predicted mean total: {cc.predicted_mean_total:.4f}")
        print(f"  Actual mean total:    {cc.actual_mean_total:.4f}")
        print(f"  Bias:                 {cc.bias:+.4f} ({abs(cc.bias)/cc.actual_mean_total*100:.1f}%)")

    if report.card_calibration:
        kc = report.card_calibration
        print(f"\nBooking Points (n={kc.n_predictions}):")
        print(f"  Predicted mean total: {kc.predicted_mean_total:.4f}")
        print(f"  Actual mean total:    {kc.actual_mean_total:.4f}")
        print(f"  Bias:                 {kc.bias:+.4f} ({abs(kc.bias)/kc.actual_mean_total*100:.1f}%)")

    # Also report mean total in raw card units (yellows + reds)
    heldout_mask = df["season"].isin(config.held_out_seasons)
    ho_df = df[heldout_mask]
    if "home_booking_points" in ho_df.columns:
        raw_bp_total = (ho_df["home_booking_points"] + ho_df["away_booking_points"]).dropna()
        print(f"\n  Raw booking point totals (held-out):")
        print(f"    Mean: {raw_bp_total.mean():.2f}")
        print(f"    Std:  {raw_bp_total.std():.2f}")
        print(f"    Min:  {raw_bp_total.min()}")
        print(f"    Max:  {raw_bp_total.max()}")

    # ================================================================
    # PART 3: Brier scores for specific O/U lines
    # ================================================================
    section("PART 3: Brier Scores for Specific O/U Lines")

    if report.corner_metrics:
        cm = report.corner_metrics
        print(f"\nCorner O/U Brier scores (model):")
        for line in sorted(cm.per_line_brier.keys()):
            print(f"  O/U {line:5.1f}: {cm.per_line_brier[line]:.6f}")

        # Reconstruct CountPredictions for baseline comparison
        from services.engine.backtest.count_types import CountPrediction
        corner_pred_objects = [
            CountPrediction(**{k: v for k, v in p.items()})
            for p in report.corner_predictions
        ]

        # Baseline Brier for specific lines
        print(f"\n  Model vs Base-rate baseline:")
        for target_line in [9.5, 10.5]:
            if target_line in cm.per_line_brier:
                model_brier = cm.per_line_brier[target_line]
                base_brier = _compute_baseline_brier(corner_pred_objects, [target_line])
                delta = model_brier - base_brier
                pct = delta / base_brier * 100 if base_brier > 0 else 0
                print(f"    O/U {target_line}: model={model_brier:.6f}  "
                      f"baseline={base_brier:.6f}  delta={delta:+.6f} ({pct:+.1f}%)")

    if report.card_metrics:
        km = report.card_metrics
        print(f"\nCard/Booking Point O/U Brier scores (model):")
        for line in sorted(km.per_line_brier.keys()):
            print(f"  O/U {line:5.1f}: {km.per_line_brier[line]:.6f}")

        card_pred_objects = [
            CountPrediction(**{k: v for k, v in p.items()})
            for p in report.card_predictions
        ]

        # For cards, check O/U lines that make sense for booking points
        # Booking points are 10*Y + 25*R, so typical total is 30-60
        print(f"\n  Model vs Base-rate baseline:")
        for target_line in [30.5, 40.5]:
            if target_line in km.per_line_brier:
                model_brier = km.per_line_brier[target_line]
                base_brier = _compute_baseline_brier(card_pred_objects, [target_line])
                delta = model_brier - base_brier
                pct = delta / base_brier * 100 if base_brier > 0 else 0
                print(f"    O/U {target_line}: model={model_brier:.6f}  "
                      f"baseline={base_brier:.6f}  delta={delta:+.6f} ({pct:+.1f}%)")

    # ================================================================
    # PART 5: Capability flag verification
    # ================================================================
    section("PART 5: Capability Flag Verification")

    print(f"\n--- E0 (has_corners=True, has_cards=True) ---")
    print(f"  Corner predictions: {len(report.corner_predictions)}")
    print(f"  Card predictions:   {len(report.card_predictions)}")
    print(f"  Corner gate:        {'PASS' if report.corner_gate_passed else 'FAIL'}")
    print(f"  Card gate:          {'PASS' if report.card_gate_passed else 'FAIL'}")

    # Run a mini backtest with league_code=None to verify suppression
    print(f"\n--- No league_code (should suppress count models) ---")
    config_no_count = BacktestConfig(
        held_out_seasons=league.test_seasons,
        training_start_season=league.training_start_season,
        xi=league.xi,
        rho_bounds=league.rho_bounds,
        max_goals=league.max_goals,
        refit_step="per_date",
        seed=42,
        # league_code=None (default)
    )
    report_no_count = run_backtest(df, config_no_count)
    print(f"  Corner predictions: {len(report_no_count.corner_predictions)}")
    print(f"  Card predictions:   {len(report_no_count.card_predictions)}")
    print(f"  Corner metrics:     {report_no_count.corner_metrics}")
    print(f"  Card metrics:       {report_no_count.card_metrics}")

    if (len(report_no_count.corner_predictions) == 0
            and len(report_no_count.card_predictions) == 0):
        print("  CONFIRMED: Count models suppressed when league_code is None")
    else:
        print("  PROBLEM: Count models should be suppressed!")

    # ================================================================
    # Gate summary
    # ================================================================
    section("GATE SUMMARY")

    print(f"\nGoals Gate: {'PASS' if report.gate_passed else 'FAIL'}")
    for d in report.gate_details:
        status = "PASS" if d.passed else "FAIL"
        print(f"  [{status}] {d.name}: {d.message}")

    if report.corner_gate_details:
        print(f"\nCorners Gate: {'PASS' if report.corner_gate_passed else 'FAIL'}")
        for d in report.corner_gate_details:
            status = "PASS" if d.passed else "FAIL"
            print(f"  [{status}] {d.name}: {d.message}")

    if report.card_gate_details:
        print(f"\nCards Gate: {'PASS' if report.card_gate_passed else 'FAIL'}")
        for d in report.card_gate_details:
            status = "PASS" if d.passed else "FAIL"
            print(f"  [{status}] {d.name}: {d.message}")

    print(f"\nTotal elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
