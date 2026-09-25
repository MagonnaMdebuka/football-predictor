#!/usr/bin/env python3
"""Run walk-forward backtest and report total-corner model diagnostics.

Reports:
1. Implied vs observed total corner variance (convolution vs total model)
2. Decile calibration table with slope
3. Gate Brier on [8.5-12.5] vs base-rate AND simple rate-based model
4. Home/away booking-point correlation and variance comparison
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, poisson as poisson_dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.backtest.count_gate import _compute_baseline_brier
from services.engine.backtest.count_types import CountPrediction
from services.engine.backtest.harness import run_backtest
from services.engine.backtest.types import BacktestConfig
from services.engine.config import get_league_config
from services.engine.markets.counts import CORNER_GATE_LINES
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

    pred_home = np.array([p.mu_home for p in corner_preds])
    pred_away = np.array([p.mu_away for p in corner_preds])
    actual_home = np.array([p.actual_home for p in corner_preds])
    actual_away = np.array([p.actual_away for p in corner_preds])
    actual_total = actual_home + actual_away
    alphas = np.array([p.alpha for p in corner_preds])

    # Check if mu_total is populated (new total model)
    has_total = corner_preds[0].mu_total is not None
    if has_total:
        mu_total = np.array([p.mu_total for p in corner_preds])
        alpha_total = np.array([p.alpha_total for p in corner_preds])
    else:
        mu_total = pred_home + pred_away
        alpha_total = alphas

    # ================================================================
    # 1. Implied vs observed total corner variance
    # ================================================================
    print("\n" + "=" * 70)
    print("  1. IMPLIED vs OBSERVED TOTAL CORNER VARIANCE")
    print("=" * 70)

    obs_var = float(actual_total.var())
    obs_mean = float(actual_total.mean())

    # Convolution model: Var(H+A) = Var(H) + Var(A) assuming independence
    conv_var_per_match = (
        (pred_home + alphas * pred_home ** 2) +
        (pred_away + alphas * pred_away ** 2)
    )
    conv_var = float(conv_var_per_match.mean())

    # Total model: Var(T) = mu_t + alpha_t * mu_t^2
    total_var_per_match = mu_total + alpha_total * mu_total ** 2
    total_var = float(total_var_per_match.mean())

    print(f"\n  Observed mean total:    {obs_mean:.3f}")
    print(f"  Observed Var(total):   {obs_var:.3f}")
    print(f"  Convolution implied:   {conv_var:.3f}  (ratio: {conv_var / obs_var:.3f})")
    print(f"  Total model implied:   {total_var:.3f}  (ratio: {total_var / obs_var:.3f})")

    # ================================================================
    # 2. Decile calibration
    # ================================================================
    print("\n" + "=" * 70)
    print("  2. DECILE CALIBRATION (total model predicted total)")
    print("=" * 70)

    pred_total_for_decile = mu_total if has_total else (pred_home + pred_away)
    order = np.argsort(pred_total_for_decile)
    decile_size = n // 10
    remainder = n % 10

    print(f"\n  {'Decile':>7}  {'N':>4}  {'Pred Mean':>10}  {'Actual Mean':>11}")
    print("  " + "-" * 40)

    dec_pred = []
    dec_actual = []
    start = 0
    for d in range(10):
        size = decile_size + (1 if d < remainder else 0)
        idx = order[start:start + size]
        pm = float(pred_total_for_decile[idx].mean())
        am = float(actual_total[idx].mean())
        dec_pred.append(pm)
        dec_actual.append(am)
        print(f"  {d + 1:>7d}  {size:>4d}  {pm:>10.2f}  {am:>11.2f}")
        start += size

    slope = np.polyfit(dec_pred, dec_actual, 1)[0]
    dec_corr, _ = pearsonr(dec_pred, dec_actual)
    print(f"\n  Slope: {slope:.3f}   r: {dec_corr:.4f}")

    # ================================================================
    # 3. Gate Brier on [8.5-12.5]
    # ================================================================
    print("\n" + "=" * 70)
    print("  3. GATE BRIER on [8.5-12.5]")
    print("=" * 70)

    # Model Brier (from walk-forward predictions — uses total model for match O/U)
    model_briers = {}
    for line in CORNER_GATE_LINES:
        brier_sum = 0.0
        for p in corner_preds:
            at = p.actual_home + p.actual_away
            indicator = 1.0 if at > line else 0.0
            for ou in p.match_over_under:
                if ou["line"] == line:
                    brier_sum += (ou["over"] - indicator) ** 2
                    break
        model_briers[line] = brier_sum / n

    # Base-rate Brier
    base_brier = _compute_baseline_brier(corner_preds, list(CORNER_GATE_LINES))

    # Simple rate-based Brier
    held_out_seasons = set(config.held_out_seasons)
    train_df = df[~df["season"].isin(held_out_seasons)].dropna(
        subset=["home_corners", "away_corners"]
    )
    team_cf = {}
    team_ca = {}
    for team in set(train_df["home_team"]) | set(train_df["away_team"]):
        hm = train_df[train_df["home_team"] == team]
        am = train_df[train_df["away_team"] == team]
        cf = list(hm["home_corners"].dropna()) + list(am["away_corners"].dropna())
        ca = list(hm["away_corners"].dropna()) + list(am["home_corners"].dropna())
        if cf:
            team_cf[team] = float(np.mean(cf))
        if ca:
            team_ca[team] = float(np.mean(ca))
    league_mean = float(np.mean(list(team_cf.values())))

    held_df = df[df["season"].isin(held_out_seasons)].dropna(
        subset=["home_corners", "away_corners"]
    )
    simple_pred = []
    simple_actual_arr = []
    for _, row in held_df.iterrows():
        hf = team_cf.get(row["home_team"], league_mean)
        ha = team_ca.get(row["home_team"], league_mean)
        af = team_cf.get(row["away_team"], league_mean)
        aa = team_ca.get(row["away_team"], league_mean)
        simple_pred.append((hf + aa) / 2 + (af + ha) / 2)
        simple_actual_arr.append(int(row["home_corners"]) + int(row["away_corners"]))
    simple_pred = np.array(simple_pred)
    simple_actual_arr = np.array(simple_actual_arr)
    n_simple = len(simple_pred)

    simple_briers = {}
    for line in CORNER_GATE_LINES:
        k = int(line)
        ss = 0.0
        for i in range(n_simple):
            p_over = 1.0 - float(poisson_dist.cdf(k, simple_pred[i]))
            indicator = 1.0 if simple_actual_arr[i] > line else 0.0
            ss += (p_over - indicator) ** 2
        simple_briers[line] = ss / n_simple

    model_gate = np.mean(list(model_briers.values()))
    simple_gate = np.mean(list(simple_briers.values()))

    print(f"\n  {'Line':>6}  {'Model':>10}  {'Base':>10}  {'Simple':>10}")
    print("  " + "-" * 42)
    for line in CORNER_GATE_LINES:
        bl = _compute_baseline_brier(corner_preds, [line])
        print(f"  {line:>6.1f}  {model_briers[line]:>10.6f}  {bl:>10.6f}  "
              f"{simple_briers[line]:>10.6f}")

    print(f"\n  Gate mean:")
    print(f"    Model:  {model_gate:.6f}  ({(model_gate - base_brier) / base_brier * 100:+.2f}% vs base)")
    print(f"    Simple: {simple_gate:.6f}  ({(simple_gate - base_brier) / base_brier * 100:+.2f}% vs base)")
    print(f"    Base:   {base_brier:.6f}")

    # ================================================================
    # 4. Booking-point correlation and variance
    # ================================================================
    print("\n" + "=" * 70)
    print("  4. BOOKING-POINT HOME/AWAY CORRELATION & VARIANCE")
    print("=" * 70)

    bp_df = df.dropna(subset=["home_booking_points", "away_booking_points"])
    if len(bp_df) > 10:
        home_bp = bp_df["home_booking_points"].values.astype(float)
        away_bp = bp_df["away_booking_points"].values.astype(float)
        total_bp = home_bp + away_bp
        r_bp, p_bp = pearsonr(home_bp, away_bp)

        obs_var_bp = float(total_bp.var())
        obs_var_h_bp = float(home_bp.var())
        obs_var_a_bp = float(away_bp.var())
        indep_var_bp = obs_var_h_bp + obs_var_a_bp

        print(f"\n  N matches:            {len(bp_df)}")
        print(f"  Pearson r(H,A):       {r_bp:.4f}  p={p_bp:.6f}")
        print(f"  Var(home BP):         {obs_var_h_bp:.1f}")
        print(f"  Var(away BP):         {obs_var_a_bp:.1f}")
        print(f"  Var(total BP):        {obs_var_bp:.1f}")
        print(f"  Independence implied: {indep_var_bp:.1f}  (ratio: {indep_var_bp / obs_var_bp:.3f})")

        card_preds = [CountPrediction(**p) for p in report.card_predictions]
        if card_preds:
            card_mu_h = np.array([p.mu_home for p in card_preds])
            card_mu_a = np.array([p.mu_away for p in card_preds])
            card_alphas = np.array([p.alpha for p in card_preds])
            card_conv_var = float(
                ((card_mu_h + card_alphas * card_mu_h ** 2) +
                 (card_mu_a + card_alphas * card_mu_a ** 2)).mean()
            )
            print(f"  Model conv implied:   {card_conv_var:.1f}  (ratio: {card_conv_var / obs_var_bp:.3f})")
    else:
        print("  Insufficient booking-point data")


if __name__ == "__main__":
    main()
