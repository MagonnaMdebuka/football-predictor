"""Sweep xi on a validation slice and confirm on the test set.

Usage: python -u scripts/tune_xi.py

Validation slice: 2022-23, 2023-24  (hyperparameter selection)
Test slice:       2024-25, 2025-26  (final confirmation, never used for selection)

Reports RPS and log loss per xi value, picks the best by log loss, then runs
one final confirmation on the test slice and compares against xi=0.0065.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.backtest.harness import run_backtest
from services.engine.backtest.types import BacktestConfig
from services.engine.models.decay import half_life_days


def load_all_seasons() -> pd.DataFrame:
    """Load all raw CSV seasons into a single DataFrame."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
    )
    import json

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


XI_CANDIDATES = [0.0, 0.002, 0.004, 0.0065, 0.010, 0.015, 0.020, 0.030]

VALIDATION_SEASONS = ("2022-23", "2023-24")
TEST_SEASONS = ("2024-25", "2025-26")
TRAINING_START = "2019-20"
BASELINE_XI = 0.0065


def run_sweep(df: pd.DataFrame) -> list[dict]:
    """Run backtest for each xi candidate on the validation slice."""
    results = []
    for xi in XI_CANDIDATES:
        config = BacktestConfig(
            held_out_seasons=VALIDATION_SEASONS,
            training_start_season=TRAINING_START,
            xi=xi,
            refit_step="per_date",
            seed=42,
        )
        t0 = time.time()
        report = run_backtest(df, config)
        elapsed = time.time() - t0

        hl = f"{half_life_days(xi):.0f}d" if xi > 0 else "inf"
        results.append({
            "xi": xi,
            "half_life": hl,
            "rps": report.combined.model.rps,
            "log_loss": report.combined.model.log_loss,
            "hit_rate": report.combined.model.hit_rate,
            "n": report.combined.model.n_matches,
            "elapsed": elapsed,
        })
        print(
            f"  xi={xi:.4f}  HL={hl:>6s}  "
            f"RPS={report.combined.model.rps:.6f}  "
            f"LL={report.combined.model.log_loss:.6f}  "
            f"HR={report.combined.model.hit_rate:.3f}  "
            f"({elapsed:.0f}s)",
            flush=True,
        )
    return results


def main():
    t0_total = time.time()
    print("Loading data...", flush=True)
    df = load_all_seasons()
    print(f"Loaded {len(df)} matches across {df['season'].nunique()} seasons\n", flush=True)

    # --- Validation sweep ---
    print(f"{'='*70}")
    print(f"VALIDATION SWEEP  (held-out: {', '.join(VALIDATION_SEASONS)})")
    print(f"{'='*70}")
    results = run_sweep(df)

    # Pick best by log loss
    best = min(results, key=lambda r: r["log_loss"])
    best_xi = best["xi"]

    print(f"\n{'='*70}")
    print("SWEEP RESULTS (sorted by log loss)")
    print(f"{'='*70}")
    print(f"  {'xi':>6s}  {'HL':>6s}  {'RPS':>10s}  {'LogLoss':>10s}  {'HitRate':>7s}  {'best':>4s}")
    print(f"  {'------':>6s}  {'------':>6s}  {'----------':>10s}  {'----------':>10s}  {'-------':>7s}  {'----':>4s}")
    for r in sorted(results, key=lambda r: r["log_loss"]):
        marker = " <--" if r["xi"] == best_xi else ""
        print(
            f"  {r['xi']:>6.4f}  {r['half_life']:>6s}  "
            f"{r['rps']:>10.6f}  {r['log_loss']:>10.6f}  "
            f"{r['hit_rate']:>7.3f}  {marker}"
        )

    print(f"\nBest xi by log loss: {best_xi:.4f} (half-life: {best['half_life']})")

    # --- Final confirmation on test set ---
    print(f"\n{'='*70}")
    print(f"FINAL CONFIRMATION  (held-out: {', '.join(TEST_SEASONS)})")
    print(f"{'='*70}")

    configs = {}
    for xi_val, label in [(best_xi, "best"), (BASELINE_XI, "baseline (0.0065)")]:
        config = BacktestConfig(
            held_out_seasons=TEST_SEASONS,
            training_start_season=TRAINING_START,
            xi=xi_val,
            refit_step="per_date",
            seed=42,
        )
        t0 = time.time()
        report = run_backtest(df, config)
        elapsed = time.time() - t0
        configs[label] = {
            "xi": xi_val,
            "rps": report.combined.model.rps,
            "log_loss": report.combined.model.log_loss,
            "hit_rate": report.combined.model.hit_rate,
            "n": report.combined.model.n_matches,
            "gate_passed": report.gate_passed,
            "elapsed": elapsed,
        }
        hl = f"{half_life_days(xi_val):.0f}d" if xi_val > 0 else "inf"
        print(
            f"  {label}: xi={xi_val:.4f}  HL={hl}  "
            f"RPS={report.combined.model.rps:.6f}  "
            f"LL={report.combined.model.log_loss:.6f}  "
            f"HR={report.combined.model.hit_rate:.3f}  "
            f"Gate={'PASS' if report.gate_passed else 'FAIL'}  ({elapsed:.0f}s)",
            flush=True,
        )

    # Skip the duplicate if best_xi == BASELINE_XI
    if best_xi == BASELINE_XI:
        print("\nBest xi equals the baseline — no change needed.")
    else:
        b = configs["best"]
        bl = configs["baseline (0.0065)"]
        rps_delta = b["rps"] - bl["rps"]
        ll_delta = b["log_loss"] - bl["log_loss"]
        print(f"\n  Delta (best - baseline):  RPS={rps_delta:+.6f}  LL={ll_delta:+.6f}")
        if ll_delta < 0:
            print(f"\n  --> UPDATE league_defaults.py: E0 xi = {best_xi}")
        else:
            print(f"\n  --> KEEP xi = {BASELINE_XI} (best on validation did not improve test)")

    total = time.time() - t0_total
    print(f"\nTotal elapsed: {total:.0f}s")


if __name__ == "__main__":
    main()
