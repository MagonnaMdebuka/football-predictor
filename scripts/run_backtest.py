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
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def progress(step: int, total: int, refit_date) -> None:
    print(f"  Refit {step}/{total}  cutoff={refit_date.date()}", flush=True)


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    df = load_all_seasons()
    print(f"Loaded {len(df)} matches across {df['season'].nunique()} seasons", flush=True)

    config = BacktestConfig(
        held_out_seasons=("2024-25", "2025-26"),
        training_start_season="2019-20",
        xi=0.0065,
        refit_step="per_date",
        seed=42,
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
    if c.bookmaker:
        print(
            f"  Bookmaker:RPS={c.bookmaker.rps:.6f}  LL={c.bookmaker.log_loss:.6f}"
        )

    print("\n=== Gate ===")
    for d in report.gate_details:
        status = "PASS" if d.passed else "FAIL"
        print(f"  [{status}] {d.name}: {d.message}")
    print(f"\nGate: {'PASS' if report.gate_passed else 'FAIL'}")


if __name__ == "__main__":
    main()
