"""Phase 8 diagnostic: run backtest with halftime enabled and report metrics.

Usage: python scripts/halftime_diagnostic.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.engine.backtest.halftime_gate import (
    _compute_ht_baselines,
    _compute_ht_metrics,
    _compute_htft_rps,
    _compute_uniform_htft_rps,
    _sh_result,
    run_halftime_gate_checks,
)
from services.engine.backtest.harness import run_backtest
from services.engine.backtest.metrics import (
    log_loss,
    ranked_probability_score,
)
from services.engine.backtest.types import BacktestConfig


def _infer_season(dt) -> str:
    """Infer season string from a date."""
    if dt.month >= 7:
        return f"{dt.year}-{str(dt.year + 1)[-2:]}"
    return f"{dt.year - 1}-{str(dt.year)[-2:]}"


def _load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and normalise columns for the backtest harness."""
    import json
    raw = pd.read_csv(path)

    if "Date" in raw.columns:
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], dayfirst=True),
            "home_team": raw.get("HomeTeam", raw.get("HT")),
            "away_team": raw.get("AwayTeam", raw.get("AT")),
            "home_goals": raw.get("FTHG", raw.get("HG")),
            "away_goals": raw.get("FTAG", raw.get("AG")),
            "ftr": raw.get("FTR", raw.get("Res")),
        })
    else:
        raise ValueError("CSV must have 'Date' column")

    if "season" not in raw.columns and "Season" not in raw.columns:
        df["season"] = df["date"].apply(_infer_season)
    else:
        df["season"] = raw.get("season", raw.get("Season"))

    # HT goals
    if "HTHG" in raw.columns and "HTAG" in raw.columns:
        df["ht_home_goals"] = pd.to_numeric(raw["HTHG"], errors="coerce")
        df["ht_away_goals"] = pd.to_numeric(raw["HTAG"], errors="coerce")

    # Count columns
    count_cols = {
        "HC": "home_corners", "AC": "away_corners",
        "HY": "home_yellows", "AY": "away_yellows",
        "HR": "home_reds", "AR": "away_reds",
    }
    for csv_col, df_col in count_cols.items():
        if csv_col in raw.columns:
            df[df_col] = pd.to_numeric(raw[csv_col], errors="coerce")

    # Bookmaker odds
    odds_cols = ["PSCH", "PSCD", "PSCA", "AvgCH", "AvgCD", "AvgCA"]
    available_odds = [c for c in odds_cols if c in raw.columns]
    if available_odds:
        df["source_row_raw"] = raw[available_odds].apply(
            lambda row: json.dumps({k: v for k, v in row.items() if pd.notna(v)}),
            axis=1,
        )

    return df


# ── Load and concatenate CSV files ──────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Use files that have HTHG column (exclude 202021 which is broken HTML)
CSV_FILES = sorted([
    f for f in DATA_DIR.glob("E0_20*.csv")
    if "202021" not in f.name  # broken file
])

print("Loading CSVs...")
frames = []
for f in CSV_FILES:
    try:
        one = _load_csv(str(f))
        print(f"  {f.name}: {len(one)} rows, "
              f"HTHG={'yes' if 'ht_home_goals' in one.columns else 'NO'}")
        frames.append(one)
    except Exception as e:
        print(f"  {f.name}: SKIP ({e})")

df = pd.concat(frames, ignore_index=True)
df = df.sort_values("date").reset_index(drop=True)
print(f"\nTotal: {len(df)} matches, "
      f"seasons: {sorted(df['season'].unique())}")

# Check HT data availability
ht_available = df["ht_home_goals"].notna().sum()
print(f"HT goals available: {ht_available}/{len(df)} "
      f"({100*ht_available/len(df):.1f}%)")


# ── Run backtest ────────────────────────────────────────────────────────
config = BacktestConfig(
    held_out_seasons=("2024-25", "2025-26"),
    training_start_season="2019-20",
    xi=0.0065,
    refit_step="per_date",
    league_code="E0",
)

print("\nRunning backtest (this may take several minutes)...")
t0 = time.time()


def progress(refit_idx, n_refits, refit_date):
    if refit_idx % 20 == 0 or refit_idx == n_refits:
        elapsed = time.time() - t0
        print(f"  Refit {refit_idx}/{n_refits} ({elapsed:.0f}s)")


report = run_backtest(df, config, progress_callback=progress)
elapsed = time.time() - t0
print(f"Backtest complete in {elapsed:.0f}s")

# ── Extract halftime predictions ────────────────────────────────────────
ht_preds = report.halftime_predictions
n_ht = len(ht_preds)
print(f"\nHalftime predictions: {n_ht}")

if n_ht == 0:
    print("ERROR: No halftime predictions generated!")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════
# 1. FITTED FIRST-HALF GOAL SHARE
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. FITTED FIRST-HALF GOAL SHARE")
print("=" * 70)

ht_goals = sum(p.ht_home_goals + p.ht_away_goals for p in ht_preds)
ft_goals = sum(
    (p.ht_home_goals + p.sh_home_goals) + (p.ht_away_goals + p.sh_away_goals)
    for p in ht_preds
)
ht_share = ht_goals / ft_goals if ft_goals > 0 else 0
print(f"HT goals: {ht_goals}, FT goals: {ft_goals}")
print(f"First-half goal share: {ht_share:.4f} (expect ~0.45)")

# Model lambda share
ht_lam_total = sum(p.ht_lambda_home + p.ht_lambda_away for p in ht_preds)
sh_lam_total = sum(p.sh_lambda_home + p.sh_lambda_away for p in ht_preds)
fitted_share = ht_lam_total / (ht_lam_total + sh_lam_total)
print(f"Fitted lambda share (HT / (HT+2H)): {fitted_share:.4f}")
print(f"Mean HT lambda: {ht_lam_total/n_ht:.4f}")
print(f"Mean 2H lambda: {sh_lam_total/n_ht:.4f}")

# ════════════════════════════════════════════════════════════════════════
# 2. HT 1X2: RPS AND LOG LOSS VS BASE-RATE
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. HT 1X2: RPS AND LOG LOSS vs BASE-RATE BASELINE")
print("=" * 70)

# Split by season (infer from date)
season_preds: dict[str, list] = {}
for p in ht_preds:
    s = _infer_season(pd.Timestamp(p.date))
    season_preds.setdefault(s, []).append(p)

for label, preds in list(season_preds.items()) + [("COMBINED", ht_preds)]:
    n = len(preds)
    ht_actual = np.array([p.ht_result for p in preds])

    # Model
    ht_p_home = np.array([p.ht_model_home for p in preds], dtype=np.float64)
    ht_p_draw = np.array([p.ht_model_draw for p in preds], dtype=np.float64)
    ht_p_away = np.array([p.ht_model_away for p in preds], dtype=np.float64)
    model_rps = ranked_probability_score(ht_p_home, ht_p_draw, ht_p_away, ht_actual)
    model_ll = log_loss(ht_p_home, ht_p_draw, ht_p_away, ht_actual)

    # Base-rate
    ht_br, _ = _compute_ht_baselines(preds)
    br_rps = ranked_probability_score(
        np.full(n, ht_br[0]), np.full(n, ht_br[1]), np.full(n, ht_br[2]),
        ht_actual,
    )
    br_ll = log_loss(
        np.full(n, ht_br[0]), np.full(n, ht_br[1]), np.full(n, ht_br[2]),
        ht_actual,
    )

    # 2H model
    sh_actual = np.array([_sh_result(p) for p in preds])
    sh_p_home = np.array([p.sh_model_home for p in preds], dtype=np.float64)
    sh_p_draw = np.array([p.sh_model_draw for p in preds], dtype=np.float64)
    sh_p_away = np.array([p.sh_model_away for p in preds], dtype=np.float64)
    sh_model_rps = ranked_probability_score(sh_p_home, sh_p_draw, sh_p_away, sh_actual)

    _, sh_br = _compute_ht_baselines(preds)
    sh_br_rps = ranked_probability_score(
        np.full(n, sh_br[0]), np.full(n, sh_br[1]), np.full(n, sh_br[2]),
        sh_actual,
    )

    # HT draw rate
    ht_draw_rate = sum(1 for p in preds if p.ht_result == "D") / n

    rps_delta = (model_rps - br_rps) / br_rps * 100
    ll_delta = (model_ll - br_ll) / br_ll * 100

    print(f"\n  {label} (n={n}, HT draw rate={ht_draw_rate:.3f})")
    print(f"    HT 1X2  Model RPS={model_rps:.6f}  Base-rate RPS={br_rps:.6f}"
          f"  delta={rps_delta:+.2f}%")
    print(f"    HT 1X2  Model LL ={model_ll:.6f}  Base-rate LL ={br_ll:.6f}"
          f"  delta={ll_delta:+.2f}%")
    print(f"    2H 1X2  Model RPS={sh_model_rps:.6f}  Base-rate RPS={sh_br_rps:.6f}"
          f"  delta={(sh_model_rps-sh_br_rps)/sh_br_rps*100:+.2f}%")

# ════════════════════════════════════════════════════════════════════════
# 3. HT/FT 9-OUTCOME: LOG LOSS VS UNIFORM
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. HT/FT 9-OUTCOME: LOG LOSS vs UNIFORM (1/9 = 2.197)")
print("=" * 70)

outcomes = ["HH", "HD", "HA", "DH", "DD", "DA", "AH", "AD", "AA"]
eps = 1e-15

# Model log loss
htft_ll_sum = 0.0
for p in ht_preds:
    actual_key = p.ht_result + p.ft_result
    prob = max(p.htft_probs.get(actual_key, eps), eps)
    htft_ll_sum -= math.log(prob)
htft_model_ll = htft_ll_sum / n_ht

# Uniform log loss
htft_uniform_ll = -math.log(1.0 / 9.0)  # = ln(9) ≈ 2.197

# RPS comparison
htft_model_rps = _compute_htft_rps(ht_preds)
htft_uniform_rps = _compute_uniform_htft_rps(ht_preds)

print(f"  Model log loss:   {htft_model_ll:.4f}")
print(f"  Uniform log loss: {htft_uniform_ll:.4f}")
print(f"  Delta:            {htft_model_ll - htft_uniform_ll:+.4f}"
      f"  ({(htft_model_ll - htft_uniform_ll)/htft_uniform_ll*100:+.2f}%)")
print(f"  Model RPS:        {htft_model_rps:.6f}")
print(f"  Uniform RPS:      {htft_uniform_rps:.6f}")

# Outcome distribution
print(f"\n  Outcome distribution (predicted vs actual):")
actual_counts = {}
pred_means = {}
for o in outcomes:
    actual_counts[o] = sum(1 for p in ht_preds if p.ht_result + p.ft_result == o)
    pred_means[o] = np.mean([p.htft_probs.get(o, 0) for p in ht_preds])

print(f"  {'Outcome':<8} {'Predicted':>10} {'Actual':>10} {'Actual%':>8}")
for o in outcomes:
    print(f"  {o:<8} {pred_means[o]:>10.4f} {actual_counts[o]:>10}"
          f" {actual_counts[o]/n_ht*100:>7.1f}%")

# ════════════════════════════════════════════════════════════════════════
# 4. FIRST GOAL TIMING
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. FIRST GOAL TIMING")
print("=" * 70)

print("\n  Note: backtest CSV data does not contain minute-by-minute goal")
print("  times, so we cannot compute observed frequencies per bucket.")
print("  Reporting model predictions and analytical expectations instead.")
print()

# Sample a few predictions and show their first-goal timing
from services.engine.markets.halftime import first_goal_timing

# Compute mean predicted P(before X) across all predictions
bucket_probs = {15: [], 30: [], 45: []}
for p in ht_preds:
    # Use FT lambdas (HT lambda + 2H lambda approximates FT lambda)
    lam_h = p.ht_lambda_home + p.sh_lambda_home
    lam_a = p.ht_lambda_away + p.sh_lambda_away
    timing = first_goal_timing(lam_h, lam_a)
    for m in timing:
        if m["selection"].startswith("before_"):
            minute = int(m["selection"].split("_")[1])
            bucket_probs[minute].append(m["probability"])

# Observed: we can compute P(no goals at HT) and P(0-0 at FT)
p_no_ht_goals = sum(
    1 for p in ht_preds
    if p.ht_home_goals + p.ht_away_goals == 0
) / n_ht
p_no_ft_goals = sum(
    1 for p in ht_preds
    if (p.ht_home_goals + p.sh_home_goals + p.ht_away_goals + p.sh_away_goals) == 0
) / n_ht

print(f"  {'Bucket':<15} {'Mean P(before)':>15} {'Std':>8}")
for minute in [15, 30, 45]:
    mean_p = np.mean(bucket_probs[minute])
    std_p = np.std(bucket_probs[minute])
    print(f"  Before {minute}'    {mean_p:>15.4f} {std_p:>8.4f}")

print(f"\n  Observed 0-0 at HT:  {p_no_ht_goals:.4f} (= P(no goal in 45 min))")
print(f"  Model P(after 45):   {1 - np.mean(bucket_probs[45]):.4f}")
print(f"  Observed 0-0 at FT:  {p_no_ft_goals:.4f} (= P(no goal in 90 min))")

# Note about injury time
total_ht = sum(p.ht_home_goals + p.ht_away_goals for p in ht_preds)
total_sh = sum(p.sh_home_goals + p.sh_away_goals for p in ht_preds)
print(f"\n  Goals per half: HT={total_ht/n_ht:.3f}, 2H={total_sh/n_ht:.3f}")
print(f"  Ratio 2H/HT = {total_sh/total_ht:.3f} (>1 = more goals in 2H,"
      " consistent with injury time)")

# ════════════════════════════════════════════════════════════════════════
# 5. GATE RESULTS
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. GATE RESULTS")
print("=" * 70)

if report.halftime_gate_passed is not None:
    for d in report.halftime_gate_details:
        status = "PASS" if d.passed else "FAIL"
        print(f"  [{status}] {d.name}: {d.message}")
    print(f"\n  Overall gate: {'PASS' if report.halftime_gate_passed else 'FAIL'}")
else:
    print("  Gate not run (no halftime predictions)")

if report.halftime_metrics is not None:
    m = report.halftime_metrics
    print(f"\n  Metrics summary:")
    print(f"    HT RPS:      {m.ht_rps:.6f}")
    print(f"    2H RPS:      {m.sh_rps:.6f}")
    print(f"    HT/FT RPS:   {m.htft_rps:.6f}")
    print(f"    HT log loss: {m.ht_log_loss:.6f}")
    print(f"    2H log loss: {m.sh_log_loss:.6f}")
    print(f"    HT hit rate: {m.ht_hit_rate:.4f}")
    print(f"    2H hit rate: {m.sh_hit_rate:.4f}")
    print(f"    n:           {m.n_predictions}")

# ════════════════════════════════════════════════════════════════════════
# 6. CONSISTENCY CHECK
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. CONSISTENCY CHECK (first prediction)")
print("=" * 70)

p0 = ht_preds[0]
ht_sum = p0.ht_model_home + p0.ht_model_draw + p0.ht_model_away
htft_sum = sum(p0.htft_probs.values())

print(f"  Match: {p0.home_team} vs {p0.away_team} ({p0.date})")
print(f"  HT lambdas: home={p0.ht_lambda_home:.4f}, away={p0.ht_lambda_away:.4f}")
print(f"  2H lambdas: home={p0.sh_lambda_home:.4f}, away={p0.sh_lambda_away:.4f}")
print(f"\n  P(HT home) = {p0.ht_model_home:.6f}")
print(f"  P(HT draw) = {p0.ht_model_draw:.6f}")
print(f"  P(HT away) = {p0.ht_model_away:.6f}")
print(f"  Sum         = {ht_sum:.6f}  {'OK' if abs(ht_sum - 1.0) < 0.01 else 'FAIL'}")
print(f"\n  HT/FT probabilities:")
for o in outcomes:
    print(f"    {o}: {p0.htft_probs.get(o, 0):.6f}")
print(f"  Sum = {htft_sum:.6f}  {'OK' if abs(htft_sum - 1.0) < 0.01 else 'FAIL'}")

# Marginal consistency: sum of H* should ≈ P(HT=H)
ht_h_marginal = sum(p0.htft_probs.get(k, 0) for k in ["HH", "HD", "HA"])
print(f"\n  Marginal check: sum(HH+HD+HA) = {ht_h_marginal:.6f}"
      f" vs P(HT=H) = {p0.ht_model_home:.6f}"
      f"  diff = {abs(ht_h_marginal - p0.ht_model_home):.6f}")

# Actual result
print(f"\n  Actual: HT {p0.ht_home_goals}-{p0.ht_away_goals} ({p0.ht_result}),"
      f" FT {p0.ht_home_goals+p0.sh_home_goals}-{p0.ht_away_goals+p0.sh_away_goals}"
      f" ({p0.ft_result})")
