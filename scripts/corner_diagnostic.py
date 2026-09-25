"""Corner model diagnostic: where is signal lost between team fits and match totals?

1. Predicted vs observed total variance + home/away correlation
2. Decile calibration: predicted total vs actual total across bins
3. Simple rate-based alternative Brier comparison
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.backtest.count_gate import _compute_baseline_brier
from services.engine.backtest.count_types import CountPrediction
from services.engine.backtest.harness import run_backtest
from services.engine.backtest.types import BacktestConfig
from services.engine.config import get_league_config
from services.engine.markets.counts import CORNER_GATE_LINES, CORNER_MATCH_LINES
from services.engine.models.negbin import negbin_pmf


def load_data():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
    )
    season_files = {
        "2019-20": "E0_1920.csv", "2020-21": "E0_2021.csv",
        "2021-22": "E0_2122.csv", "2022-23": "E0_2223.csv",
        "2023-24": "E0_2324.csv", "2024-25": "E0_2425.csv",
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
            "season": season, "home_team": raw["HomeTeam"], "away_team": raw["AwayTeam"],
            "home_goals": raw["FTHG"].astype(int), "away_goals": raw["FTAG"].astype(int),
            "ftr": raw["FTR"],
        })
        if avail:
            df["source_row_raw"] = raw[avail].apply(
                lambda row: json.dumps({k: v for k, v in row.items() if pd.notna(v)}),
                axis=1,
            )
        if "HC" in raw.columns and "AC" in raw.columns:
            df["home_corners"] = pd.to_numeric(raw["HC"], errors="coerce").astype("Int64")
            df["away_corners"] = pd.to_numeric(raw["AC"], errors="coerce").astype("Int64")
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
        if "Referee" in raw.columns:
            df["referee"] = raw["Referee"].astype(str).str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_data()
    print(f"Loaded {len(df)} matches")

    # Run backtest
    print("Running walk-forward backtest...")
    t0 = time.time()
    league = get_league_config("E0")
    config = BacktestConfig(
        held_out_seasons=league.test_seasons,
        training_start_season=league.training_start_season,
        xi=league.xi, rho_bounds=league.rho_bounds,
        max_goals=league.max_goals, refit_step="per_date",
        seed=42, league_code="E0",
    )
    report = run_backtest(df, config)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s  corners={len(report.corner_predictions)}")

    corner_preds = [CountPrediction(**p) for p in report.corner_predictions]
    n = len(corner_preds)

    # Extract arrays
    pred_home = np.array([p.mu_home for p in corner_preds])
    pred_away = np.array([p.mu_away for p in corner_preds])
    pred_total = pred_home + pred_away
    actual_home = np.array([p.actual_home for p in corner_preds])
    actual_away = np.array([p.actual_away for p in corner_preds])
    actual_total = actual_home + actual_away
    alphas = np.array([p.alpha for p in corner_preds])

    # ================================================================
    # PART 1: Variance and correlation
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Predicted vs Observed Variance + Correlation")
    print("=" * 70)

    # Predicted total variance (implied by NB2 convolution):
    # Var(H+A) = Var(H) + Var(A) if independent
    # Var_NB2(X) = mu + alpha * mu^2
    pred_var_home = pred_home + alphas * pred_home ** 2
    pred_var_away = pred_away + alphas * pred_away ** 2
    pred_var_total_if_indep = pred_var_home + pred_var_away
    mean_pred_var_total = float(pred_var_total_if_indep.mean())

    obs_var_total = float(actual_total.var())
    obs_var_home = float(actual_home.var())
    obs_var_away = float(actual_away.var())

    # Observed correlation
    corr_pearson, pval_pearson = pearsonr(actual_home, actual_away)
    corr_spearman, pval_spearman = spearmanr(actual_home, actual_away)

    # Observed Cov(H,A) and actual Var(H+A) decomposition
    obs_cov_ha = float(np.cov(actual_home, actual_away)[0, 1])
    obs_var_total_decomp = obs_var_home + obs_var_away + 2 * obs_cov_ha

    print(f"\n  Predicted (model-implied, assuming independence):")
    print(f"    Mean Var(home):  {pred_var_home.mean():.3f}")
    print(f"    Mean Var(away):  {pred_var_away.mean():.3f}")
    print(f"    Mean Var(total): {mean_pred_var_total:.3f}")
    print(f"    Mean total:      {pred_total.mean():.3f}")
    print(f"    Std of pred_total across matches: {pred_total.std():.3f}")
    print(f"\n  Observed:")
    print(f"    Var(home):   {obs_var_home:.3f}")
    print(f"    Var(away):   {obs_var_away:.3f}")
    print(f"    Cov(H,A):    {obs_cov_ha:.3f}")
    print(f"    Var(total):  {obs_var_total:.3f}  "
          f"(decomposed: {obs_var_home:.3f} + {obs_var_away:.3f} + 2*{obs_cov_ha:.3f} "
          f"= {obs_var_total_decomp:.3f})")
    print(f"    Mean total:  {actual_total.mean():.3f}")
    print(f"\n  Correlation between home and away corners:")
    print(f"    Pearson:  r={corr_pearson:.4f}  p={pval_pearson:.4f}")
    print(f"    Spearman: r={corr_spearman:.4f}  p={pval_spearman:.4f}")

    if obs_cov_ha > 0:
        missing_var = 2 * obs_cov_ha
        print(f"\n  Positive correlation adds {missing_var:.3f} to total variance.")
        print(f"  Model assumes independence => underestimates Var(total) by "
              f"{missing_var / obs_var_total * 100:.1f}%")
    elif obs_cov_ha < 0:
        print(f"\n  Negative correlation: home and away corners are substitutes.")

    # ================================================================
    # PART 2: Decile calibration
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Decile Calibration (predicted total vs actual total)")
    print("=" * 70)

    # Sort by predicted total, split into deciles
    order = np.argsort(pred_total)
    decile_size = n // 10
    remainder = n % 10

    print(f"\n  {'Decile':>7s}  {'N':>4s}  {'Pred Mean':>10s}  {'Actual Mean':>11s}  "
          f"{'Pred Range':>14s}  {'Actual Std':>10s}")
    print("  " + "-" * 68)

    decile_pred_means = []
    decile_actual_means = []

    start = 0
    for d in range(10):
        # Distribute remainder across first deciles
        size = decile_size + (1 if d < remainder else 0)
        idx = order[start:start + size]
        p_mean = float(pred_total[idx].mean())
        a_mean = float(actual_total[idx].mean())
        p_min = float(pred_total[idx].min())
        p_max = float(pred_total[idx].max())
        a_std = float(actual_total[idx].std())

        decile_pred_means.append(p_mean)
        decile_actual_means.append(a_mean)

        print(f"  {d + 1:>7d}  {size:>4d}  {p_mean:>10.2f}  {a_mean:>11.2f}  "
              f"  [{p_min:.1f}, {p_max:.1f}]  {a_std:>10.2f}")
        start += size

    # Correlation of decile means
    dec_corr, dec_pval = pearsonr(decile_pred_means, decile_actual_means)
    slope = np.polyfit(decile_pred_means, decile_actual_means, 1)[0]
    actual_range = decile_actual_means[-1] - decile_actual_means[0]

    print(f"\n  Decile mean correlation: r={dec_corr:.4f}  p={dec_pval:.4f}")
    print(f"  Regression slope (actual on predicted): {slope:.3f}")
    print(f"  Actual range across deciles: {actual_range:.2f} corners")

    if abs(dec_corr) > 0.8 and actual_range > 1.0:
        print("  => Strong monotonic relationship: team effects reach totals.")
        print("     If Brier is still poor, the distribution shape (convolution) is the issue.")
    elif abs(dec_corr) < 0.5 or actual_range < 0.5:
        print("  => Weak/flat relationship: team effects are NOT reaching totals.")
        print("     Corners may be close to unpredictable at match level.")
    else:
        print("  => Moderate relationship: some signal reaches totals but is attenuated.")

    # Also: raw match-level correlation
    match_corr, match_pval = pearsonr(pred_total, actual_total)
    match_spearman, _ = spearmanr(pred_total, actual_total)
    print(f"\n  Match-level pred vs actual total:")
    print(f"    Pearson:  r={match_corr:.4f}  p={match_pval:.6f}")
    print(f"    Spearman: r={match_spearman:.4f}")

    # ================================================================
    # PART 3: Simple rate-based alternative
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 3: Simple Rate-Based Alternative")
    print("=" * 70)

    # Build per-team corners-for and corners-against rates from training data
    # For each held-out match, use the training data available at that point
    # Simpler approach: use all pre-held-out data as a single training set
    held_out_seasons = set(config.held_out_seasons)
    train_df = df[~df["season"].isin(held_out_seasons)].dropna(
        subset=["home_corners", "away_corners"]
    )

    # Corners-for = avg corners when playing at home or away
    # Corners-against = avg corners conceded
    team_corners_for = {}
    team_corners_against = {}
    for team in set(train_df["home_team"]) | set(train_df["away_team"]):
        home_matches = train_df[train_df["home_team"] == team]
        away_matches = train_df[train_df["away_team"] == team]
        corners_for = (
            list(home_matches["home_corners"].dropna()) +
            list(away_matches["away_corners"].dropna())
        )
        corners_against = (
            list(home_matches["away_corners"].dropna()) +
            list(away_matches["home_corners"].dropna())
        )
        if corners_for:
            team_corners_for[team] = float(np.mean(corners_for))
        if corners_against:
            team_corners_against[team] = float(np.mean(corners_against))

    league_mean_for = float(np.mean(list(team_corners_for.values())))
    league_mean_against = float(np.mean(list(team_corners_against.values())))
    league_mean = (league_mean_for + league_mean_against) / 2

    # Build held-out data with simple predictions
    held_out_df = df[df["season"].isin(held_out_seasons)].dropna(
        subset=["home_corners", "away_corners"]
    )

    simple_pred_totals = []
    simple_actual_totals = []
    for _, row in held_out_df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        h_for = team_corners_for.get(ht, league_mean)
        h_against = team_corners_against.get(ht, league_mean)
        a_for = team_corners_for.get(at, league_mean)
        a_against = team_corners_against.get(at, league_mean)
        # Predicted home corners = (home_for + away_against) / 2
        pred_h = (h_for + a_against) / 2
        pred_a = (a_for + h_against) / 2
        simple_pred_totals.append(pred_h + pred_a)
        simple_actual_totals.append(int(row["home_corners"]) + int(row["away_corners"]))

    simple_pred = np.array(simple_pred_totals)
    simple_actual = np.array(simple_actual_totals)
    n_simple = len(simple_pred)

    # Simple Brier: use predicted total as Poisson lambda, compute P(over) for each line
    from scipy.stats import poisson as poisson_dist

    print(f"\n  Simple alternative: predicted total = (team_for + opponent_against) / 2")
    print(f"  Uses Poisson CDF for O/U probabilities (no NB2 convolution)")
    print(f"  Training: {len(train_df)} matches, held-out: {n_simple} matches")

    print(f"\n  {'Line':>6s}  {'Model Brier':>12s}  {'Simple Brier':>12s}  "
          f"{'Base Brier':>11s}  {'Model vs Base':>13s}  {'Simple vs Base':>14s}")
    print("  " + "-" * 75)

    for line in CORNER_GATE_LINES:
        k = int(line)

        # Model Brier (from walk-forward)
        model_sum = 0.0
        for p in corner_preds:
            at = p.actual_home + p.actual_away
            indicator = 1.0 if at > line else 0.0
            for ou in p.match_over_under:
                if ou["line"] == line:
                    model_sum += (ou["over"] - indicator) ** 2
                    break
        model_brier = model_sum / n

        # Simple Brier: P(over) = 1 - Poisson_CDF(k, lambda=pred_total)
        simple_sum = 0.0
        for i in range(n_simple):
            p_over = 1.0 - float(poisson_dist.cdf(k, simple_pred[i]))
            indicator = 1.0 if simple_actual[i] > line else 0.0
            simple_sum += (p_over - indicator) ** 2
        simple_brier = simple_sum / n_simple

        # Base-rate Brier
        base_brier = _compute_baseline_brier(corner_preds, [line])

        model_delta = (model_brier - base_brier) / base_brier * 100
        simple_delta = (simple_brier - base_brier) / base_brier * 100

        print(f"  {line:>6.1f}  {model_brier:>12.6f}  {simple_brier:>12.6f}  "
              f"{base_brier:>11.6f}  {model_delta:>+12.1f}%  {simple_delta:>+13.1f}%")

    # Gate-level comparison
    model_gate_briers = []
    simple_gate_briers = []
    for line in CORNER_GATE_LINES:
        k = int(line)
        ms = 0.0
        for p in corner_preds:
            at = p.actual_home + p.actual_away
            indicator = 1.0 if at > line else 0.0
            for ou in p.match_over_under:
                if ou["line"] == line:
                    ms += (ou["over"] - indicator) ** 2
                    break
        model_gate_briers.append(ms / n)

        ss = 0.0
        for i in range(n_simple):
            p_over = 1.0 - float(poisson_dist.cdf(k, simple_pred[i]))
            indicator = 1.0 if simple_actual[i] > line else 0.0
            ss += (p_over - indicator) ** 2
        simple_gate_briers.append(ss / n_simple)

    base_gate = _compute_baseline_brier(corner_preds, list(CORNER_GATE_LINES))
    model_gate = np.mean(model_gate_briers)
    simple_gate = np.mean(simple_gate_briers)

    print(f"\n  Gate mean (5 lines):")
    print(f"    Model:    {model_gate:.6f}  ({(model_gate - base_gate) / base_gate * 100:+.1f}% vs base)")
    print(f"    Simple:   {simple_gate:.6f}  ({(simple_gate - base_gate) / base_gate * 100:+.1f}% vs base)")
    print(f"    Base:     {base_gate:.6f}")

    # Simple alternative match-level correlation
    simple_corr, _ = pearsonr(simple_pred, simple_actual)
    print(f"\n  Simple pred vs actual total: r={simple_corr:.4f}")


if __name__ == "__main__":
    main()
