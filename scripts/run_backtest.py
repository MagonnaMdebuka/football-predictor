"""Run the full walk-forward backtest on real Premier League data.

Usage: python -u scripts/run_backtest.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.backtest.harness import run_backtest
from services.engine.backtest.report import write_report
from services.engine.backtest.types import BacktestConfig
from services.engine.config import get_league_config


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

        # Corner columns (HC = home corners, AC = away corners)
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


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    df = load_all_seasons()
    print(f"Loaded {len(df)} matches across {df['season'].nunique()} seasons", flush=True)

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

    print(
        f"Running backtest (refit_step={config.refit_step}, xi={config.xi})...",
        flush=True,
    )
    report = run_backtest(df, config, progress_callback=progress)
    elapsed = time.time() - t0

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtests"
    )
    filepath = write_report(report, output_dir)
    print(f"\nReport written to: {filepath}", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)

    # Print summary
    print(f"\n{'='*70}")
    print("BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"Predictions: {len(report.predictions)}")
    print(f"Schema: {report.schema_version}")
    print(f"Git: {report.git_commit}")

    for sm in report.seasons:
        print(f"\n--- {sm.season} ---")
        print(
            f"  Model:    RPS={sm.model.rps:.6f}  LL={sm.model.log_loss:.6f}  "
            f"HR={sm.model.hit_rate:.3f}  (n={sm.model.n_matches})"
        )
        print(
            f"  Uniform:  RPS={sm.uniform.rps:.6f}  LL={sm.uniform.log_loss:.6f}  "
            f"HR={sm.uniform.hit_rate:.3f}"
        )
        print(
            f"  BaseRate: RPS={sm.base_rate.rps:.6f}  LL={sm.base_rate.log_loss:.6f}  "
            f"HR={sm.base_rate.hit_rate:.3f}"
        )
        print(
            f"  IndepPoi: RPS={sm.independent_poisson.rps:.6f}  "
            f"LL={sm.independent_poisson.log_loss:.6f}  "
            f"HR={sm.independent_poisson.hit_rate:.3f}"
        )
        if sm.ablation:
            print(
                f"  Ablation: RPS={sm.ablation.rps:.6f}  "
                f"LL={sm.ablation.log_loss:.6f}  "
                f"HR={sm.ablation.hit_rate:.3f}"
            )
        if sm.bookmaker:
            print(
                f"  Bookmaker:RPS={sm.bookmaker.rps:.6f}  "
                f"LL={sm.bookmaker.log_loss:.6f}  "
                f"HR={sm.bookmaker.hit_rate:.3f}  "
                f"(excluded={sm.bookmaker_exclusion_count})"
            )

    c = report.combined
    print("\n=== Combined ===")
    print(
        f"  Model:    RPS={c.model.rps:.6f}  LL={c.model.log_loss:.6f}  "
        f"HR={c.model.hit_rate:.3f}  (n={c.model.n_matches})"
    )
    print(f"  Uniform:  RPS={c.uniform.rps:.6f}  LL={c.uniform.log_loss:.6f}")
    print(f"  BaseRate: RPS={c.base_rate.rps:.6f}  LL={c.base_rate.log_loss:.6f}")
    print(
        f"  IndepPoi: RPS={c.independent_poisson.rps:.6f}  "
        f"LL={c.independent_poisson.log_loss:.6f}"
    )
    if c.ablation:
        print(
            f"  Ablation: RPS={c.ablation.rps:.6f}  LL={c.ablation.log_loss:.6f}"
        )
    if c.bookmaker:
        print(
            f"  Bookmaker:RPS={c.bookmaker.rps:.6f}  LL={c.bookmaker.log_loss:.6f}"
        )

    if report.goal_calibration:
        gc = report.goal_calibration
        print(f"\n=== Goal Calibration (n={gc.n_matches}) ===")
        total_diff = gc.predicted_mean_total - gc.actual_mean_total
        o25_diff = gc.predicted_over_25_rate - gc.actual_over_25_rate
        print(
            f"  Mean total goals:  predicted={gc.predicted_mean_total:.4f}  "
            f"actual={gc.actual_mean_total:.4f}  "
            f"(diff={total_diff:+.4f})"
        )
        print(
            f"  P(over 2.5):       predicted={gc.predicted_over_25_rate:.4f}  "
            f"actual={gc.actual_over_25_rate:.4f}  "
            f"(diff={o25_diff:+.4f})"
        )

    # Count model results
    if report.corner_metrics:
        cm = report.corner_metrics
        print(f"\n=== Corners (n={cm.n_predictions}) ===")
        print(
            f"  Mean Brier: {cm.mean_brier:.6f}  "
            f"Predicted total: {cm.mean_predicted_total:.2f}  "
            f"Actual total: {cm.mean_actual_total:.2f}"
        )
        for line, brier in sorted(cm.per_line_brier.items()):
            print(f"    O/U {line}: Brier={brier:.6f}")

    if report.corner_calibration:
        cc = report.corner_calibration
        print(
            f"  Calibration: bias={cc.bias:+.4f} "
            f"(predicted={cc.predicted_mean_total:.2f}, actual={cc.actual_mean_total:.2f})"
        )

    if report.card_metrics:
        km = report.card_metrics
        print(f"\n=== Cards/Booking Points — Compound Model (n={km.n_predictions}) ===")
        print(
            f"  Mean Brier: {km.mean_brier:.6f}  "
            f"Predicted total: {km.mean_predicted_total:.2f}  "
            f"Actual total: {km.mean_actual_total:.2f}"
        )
        for line, brier in sorted(km.per_line_brier.items()):
            print(f"    O/U {line}: Brier={brier:.6f}")

        # Compound model details
        if report.card_predictions:
            alphas = [p.get("alpha_yellow") for p in report.card_predictions
                      if p.get("alpha_yellow") is not None]
            reds = [p.get("mu_red_home", 0) + p.get("mu_red_away", 0)
                    for p in report.card_predictions
                    if p.get("mu_red_home") is not None]
            if alphas:
                import numpy as _np
                print(f"  Yellow alpha: mean={_np.mean(alphas):.6f}")
            if reds:
                import numpy as _np
                print(f"  Red rate (total): mean={_np.mean(reds):.4f}")

    if report.card_calibration:
        kc = report.card_calibration
        print(
            f"  Calibration: bias={kc.bias:+.4f} "
            f"(predicted={kc.predicted_mean_total:.2f}, actual={kc.actual_mean_total:.2f})"
        )

    print("\n=== Goals Gate ===")
    for d in report.gate_details:
        status = "PASS" if d.passed else "FAIL"
        print(f"  [{status}] {d.name}: {d.message}")
    print(f"\nGoals Gate: {'PASS' if report.gate_passed else 'FAIL'}")

    if report.corner_gate_details:
        print("\n=== Corners Gate ===")
        for d in report.corner_gate_details:
            status = "PASS" if d.passed else "FAIL"
            print(f"  [{status}] {d.name}: {d.message}")
        print(f"\nCorners Gate: {'PASS' if report.corner_gate_passed else 'FAIL'}")

    if report.card_gate_details:
        print("\n=== Cards Gate ===")
        for d in report.card_gate_details:
            status = "PASS" if d.passed else "FAIL"
            print(f"  [{status}] {d.name}: {d.message}")
        print(f"\nCards Gate: {'PASS' if report.card_gate_passed else 'FAIL'}")


if __name__ == "__main__":
    main()
