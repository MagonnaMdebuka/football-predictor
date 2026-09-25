#!/usr/bin/env python3
"""Sweep shrinkage factor k for total-corner team effects.

For each k in {0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0}:
  mu_total = exp(mu + k*home_effect[h] + k*away_effect[a])

Tunes on validation seasons (2022-23, 2023-24), confirms on test seasons
(2024-25, 2025-26).

Reports: decile slope, gate Brier on [8.5-12.5], gap to base-rate and simple
rate model.
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

from services.engine.markets.counts import CORNER_GATE_LINES, CORNER_MATCH_LINES
from services.engine.models.total_count_fit import fit_total_count_model
from services.engine.models.total_count_params import pack_total, TotalCountModelParams
from services.engine.models.negbin import negbin_pmf
from services.engine.models.decay import time_weights


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
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], dayfirst=True),
            "season": season,
            "home_team": raw["HomeTeam"],
            "away_team": raw["AwayTeam"],
            "home_goals": raw["FTHG"].astype(int),
            "away_goals": raw["FTAG"].astype(int),
            "ftr": raw["FTR"],
        })
        if "HC" in raw.columns and "AC" in raw.columns:
            df["home_corners"] = pd.to_numeric(raw["HC"], errors="coerce").astype("Int64")
            df["away_corners"] = pd.to_numeric(raw["AC"], errors="coerce").astype("Int64")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _compute_match_ou(mu_total, alpha, lines):
    """Single NB2 PMF -> O/U probabilities."""
    ks = np.arange(81)
    pmf = negbin_pmf(ks, mu_total, alpha)
    cdf = np.cumsum(pmf)
    result = {}
    for line in lines:
        k = int(line)
        p_under = float(cdf[k]) if k < len(cdf) else 1.0
        result[line] = 1.0 - p_under  # P(over)
    return result


def walk_forward_with_params(df, held_out_seasons, xi=0.0065):
    """Walk-forward: fit total model per refit date, return per-prediction params.

    Returns list of dicts with: season, home_team, away_team, actual_total,
    mu, home_eff, away_eff, alpha, home_idx, away_idx (into fitted teams).
    """
    held_mask = df["season"].isin(held_out_seasons)
    held_df = df[held_mask].dropna(subset=["home_corners", "away_corners"]).copy()
    all_data = df.copy()

    pred_dates = sorted(held_df["date"].dt.normalize().unique())

    results = []
    prev_packed = None
    prev_teams = None

    for pred_date in pred_dates:
        cutoff = pd.Timestamp(pred_date)
        train_mask = all_data["date"].dt.normalize() < cutoff
        train_df = all_data[train_mask].dropna(subset=["home_corners", "away_corners"])

        if len(train_df) < 10:
            continue

        teams = sorted(
            set(train_df["home_team"].unique()) | set(train_df["away_team"].unique())
        )

        # Warm-start
        x0 = None
        if prev_packed is not None and prev_teams is not None and teams == prev_teams:
            x0 = prev_packed

        # Time weights
        match_dates = train_df["date"].values.astype("datetime64[D]")
        ref_date = cutoff.to_pydatetime().date()
        weights = time_weights(match_dates, ref_date, xi)

        fit = fit_total_count_model(
            train_df, "home_corners", "away_corners", weights=weights, x0=x0,
        )
        prev_packed = pack_total(fit.params)
        prev_teams = teams

        # Predictions for this date
        day_matches = held_df[held_df["date"].dt.normalize() == pred_date]
        team_to_idx = {t: i for i, t in enumerate(fit.params.teams)}

        for _, match in day_matches.iterrows():
            ht = match["home_team"]
            at = match["away_team"]
            hi = team_to_idx.get(ht)
            ai = team_to_idx.get(at)
            # Fallback for unknown teams: effect = 0
            h_eff = float(fit.params.home_effect[hi]) if hi is not None else 0.0
            a_eff = float(fit.params.away_effect[ai]) if ai is not None else 0.0

            results.append({
                "season": match["season"],
                "home_team": ht,
                "away_team": at,
                "actual_total": int(match["home_corners"]) + int(match["away_corners"]),
                "mu": fit.params.mu,
                "home_eff": h_eff,
                "away_eff": a_eff,
                "alpha": fit.params.alpha,
            })

    return results


def evaluate_k(predictions, k, label=""):
    """Evaluate shrinkage factor k on a set of predictions.

    Returns dict with: k, decile_slope, gate_brier, base_brier, simple_gap.
    """
    n = len(predictions)
    if n == 0:
        return None

    actual = np.array([p["actual_total"] for p in predictions])

    # Compute shrunk mu_total for each prediction
    mu_totals = np.array([
        np.exp(p["mu"] + k * p["home_eff"] + k * p["away_eff"])
        for p in predictions
    ])
    alphas = np.array([p["alpha"] for p in predictions])

    # --- Decile slope ---
    order = np.argsort(mu_totals)
    dec_size = n // 10
    rem = n % 10
    dec_pred, dec_actual = [], []
    start = 0
    for d in range(10):
        size = dec_size + (1 if d < rem else 0)
        if size == 0:
            continue
        idx = order[start:start + size]
        dec_pred.append(float(mu_totals[idx].mean()))
        dec_actual.append(float(actual[idx].mean()))
        start += size

    if len(dec_pred) >= 2:
        slope = np.polyfit(dec_pred, dec_actual, 1)[0]
        r_val, _ = pearsonr(dec_pred, dec_actual)
    else:
        slope = 0.0
        r_val = 0.0

    # --- Gate Brier ---
    model_briers = []
    base_overs = {}
    for line in CORNER_GATE_LINES:
        base_overs[line] = float(np.mean(actual > line))

    for line in CORNER_GATE_LINES:
        model_sum = 0.0
        base_sum = 0.0
        for i in range(n):
            indicator = 1.0 if actual[i] > line else 0.0
            # Model P(over)
            p_over = _compute_match_ou(mu_totals[i], alphas[i], [line])[line]
            model_sum += (p_over - indicator) ** 2
            # Base rate
            base_sum += (base_overs[line] - indicator) ** 2
        model_briers.append(model_sum / n)

    gate_brier = float(np.mean(model_briers))

    # Base-rate Brier
    base_briers = []
    for line in CORNER_GATE_LINES:
        base_over = base_overs[line]
        bs = sum((base_over - (1.0 if a > line else 0.0)) ** 2 for a in actual) / n
        base_briers.append(bs)
    base_brier = float(np.mean(base_briers))

    return {
        "k": k,
        "n": n,
        "decile_slope": round(slope, 3),
        "decile_r": round(r_val, 4),
        "gate_brier": round(gate_brier, 6),
        "base_brier": round(base_brier, 6),
        "vs_base_pct": round((gate_brier - base_brier) / base_brier * 100, 2),
    }


def compute_simple_brier(df, held_out_seasons):
    """Compute simple rate-based gate Brier for comparison."""
    held_mask = df["season"].isin(held_out_seasons)
    train_df = df[~held_mask].dropna(subset=["home_corners", "away_corners"])
    held_df = df[held_mask].dropna(subset=["home_corners", "away_corners"])

    # Per-team rates
    team_cf, team_ca = {}, {}
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

    simple_pred = []
    simple_actual = []
    for _, row in held_df.iterrows():
        hf = team_cf.get(row["home_team"], league_mean)
        ha = team_ca.get(row["home_team"], league_mean)
        af = team_cf.get(row["away_team"], league_mean)
        aa = team_ca.get(row["away_team"], league_mean)
        simple_pred.append((hf + aa) / 2 + (af + ha) / 2)
        simple_actual.append(int(row["home_corners"]) + int(row["away_corners"]))
    simple_pred = np.array(simple_pred)
    simple_actual = np.array(simple_actual)
    n = len(simple_pred)

    briers = []
    for line in CORNER_GATE_LINES:
        k = int(line)
        ss = 0.0
        for i in range(n):
            p_over = 1.0 - float(poisson_dist.cdf(k, simple_pred[i]))
            indicator = 1.0 if simple_actual[i] > line else 0.0
            ss += (p_over - indicator) ** 2
        briers.append(ss / n)
    return float(np.mean(briers))


def main():
    df = load_data()
    print(f"Loaded {len(df)} matches")

    K_VALUES = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]

    val_seasons = ("2022-23", "2023-24")
    test_seasons = ("2024-25", "2025-26")

    # Walk-forward on validation seasons
    print(f"\nFitting total model walk-forward on validation {val_seasons}...")
    t0 = time.time()
    val_preds = walk_forward_with_params(df, val_seasons)
    print(f"  {len(val_preds)} predictions in {time.time() - t0:.0f}s")

    # Walk-forward on test seasons
    print(f"\nFitting total model walk-forward on test {test_seasons}...")
    t0 = time.time()
    test_preds = walk_forward_with_params(df, test_seasons)
    print(f"  {len(test_preds)} predictions in {time.time() - t0:.0f}s")

    # Simple rate-based Brier for comparison
    simple_val = compute_simple_brier(df, val_seasons)
    simple_test = compute_simple_brier(df, test_seasons)

    # ================================================================
    # Validation sweep
    # ================================================================
    print("\n" + "=" * 80)
    print("  VALIDATION SWEEP (2022-23, 2023-24)")
    print("=" * 80)
    print(f"  Simple rate-based gate Brier: {simple_val:.6f}")
    print(f"\n  {'k':>5}  {'Slope':>6}  {'r':>7}  {'Gate Brier':>11}  "
          f"{'vs Base':>9}  {'vs Simple':>10}")
    print("  " + "-" * 56)

    val_results = []
    for k in K_VALUES:
        res = evaluate_k(val_preds, k)
        val_results.append(res)
        vs_simple = (res["gate_brier"] - simple_val) / simple_val * 100
        print(f"  {k:>5.2f}  {res['decile_slope']:>6.3f}  {res['decile_r']:>7.4f}  "
              f"{res['gate_brier']:>11.6f}  {res['vs_base_pct']:>+8.2f}%  {vs_simple:>+9.2f}%")

    # Find best k (lowest gate Brier on validation)
    best = min(val_results, key=lambda r: r["gate_brier"])
    print(f"\n  Best k on validation: {best['k']:.2f} "
          f"(Brier={best['gate_brier']:.6f}, slope={best['decile_slope']:.3f})")

    # ================================================================
    # Test confirmation
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST CONFIRMATION (2024-25, 2025-26)")
    print("=" * 80)
    print(f"  Simple rate-based gate Brier: {simple_test:.6f}")
    print(f"\n  {'k':>5}  {'Slope':>6}  {'r':>7}  {'Gate Brier':>11}  "
          f"{'vs Base':>9}  {'vs Simple':>10}")
    print("  " + "-" * 56)

    for k in K_VALUES:
        res = evaluate_k(test_preds, k)
        vs_simple = (res["gate_brier"] - simple_test) / simple_test * 100
        print(f"  {k:>5.2f}  {res['decile_slope']:>6.3f}  {res['decile_r']:>7.4f}  "
              f"{res['gate_brier']:>11.6f}  {res['vs_base_pct']:>+8.2f}%  {vs_simple:>+9.2f}%")

    # Confirm best k from validation on test
    best_test = evaluate_k(test_preds, best["k"])
    vs_simple_test = (best_test["gate_brier"] - simple_test) / simple_test * 100
    beats_simple = best_test["gate_brier"] < simple_test
    beats_base = best_test["gate_brier"] < best_test["base_brier"]
    print(f"\n  Best k={best['k']:.2f} on test: Brier={best_test['gate_brier']:.6f}  "
          f"vs base={best_test['vs_base_pct']:+.2f}%  vs simple={vs_simple_test:+.2f}%")
    print(f"  Beats base-rate: {'YES' if beats_base else 'NO'}")
    print(f"  Beats simple:    {'YES' if beats_simple else 'NO'}")

    if not beats_simple:
        print("\n  RESULT: Corners do not ship — total model with optimal shrinkage")
        print("  still fails to beat the simple rate-based model on test seasons.")
    else:
        print(f"\n  RESULT: Ship with k={best['k']:.2f}")


if __name__ == "__main__":
    main()
