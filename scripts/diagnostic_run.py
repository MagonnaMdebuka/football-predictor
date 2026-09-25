"""One-shot diagnostic: compound cards, corner shrinkage, gate analysis."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.backtest.count_gate import _compute_baseline_brier
from services.engine.backtest.count_types import CountPrediction
from services.engine.backtest.harness import run_backtest
from services.engine.backtest.types import BacktestConfig
from services.engine.config import get_league_config
from services.engine.models.count_fit import fit_count_model
from services.engine.markets.counts import (
    CORNER_MATCH_LINES, CORNER_GATE_LINES,
    BOOKING_MATCH_LINES, BOOKING_GATE_LINES,
)


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


def per_line_model_brier(preds, line):
    n = len(preds)
    brier_sum = 0.0
    actual_overs = 0
    for p in preds:
        actual_total = p.actual_home + p.actual_away
        indicator = 1.0 if actual_total > line else 0.0
        for ou in p.match_over_under:
            if ou["line"] == line:
                brier_sum += (ou["over"] - indicator) ** 2
                break
        actual_overs += int(actual_total > line)
    return brier_sum / n, actual_overs / n


def main():
    df = load_data()
    print(f"Loaded {len(df)} matches")

    # ================================================================
    # Raw data
    # ================================================================
    all_bp = pd.concat([df["home_booking_points"], df["away_booking_points"]]).dropna()
    bp_mean = float(all_bp.mean())
    bp_var = float(all_bp.var())

    all_y = pd.concat([df["home_yellows"], df["away_yellows"]]).dropna()
    y_mean = float(all_y.mean())
    y_var = float(all_y.var())

    all_r = pd.concat([df["home_reds"], df["away_reds"]]).dropna()
    r_mean = float(all_r.mean())
    r_var = float(all_r.var())

    # ================================================================
    # PART 1: Compound card model — snapshot fit
    # ================================================================
    train_mask = ~df["season"].isin(["2024-25", "2025-26"])
    train_df = df[train_mask].dropna(
        subset=["home_booking_points", "away_booking_points",
                "home_yellows", "away_yellows", "home_reds", "away_reds"]
    )

    # Direct NB2 on booking points (old approach)
    direct_fit = fit_count_model(
        train_df, "home_booking_points", "away_booking_points",
        include_referees=True, min_referee_matches=20,
    )
    direct_mu = np.exp(direct_fit.params.mu)
    direct_implied = direct_mu + direct_fit.params.alpha * direct_mu ** 2

    # Compound: yellow NB2
    yellow_fit = fit_count_model(
        train_df, "home_yellows", "away_yellows",
        include_referees=True, min_referee_matches=20,
    )
    y_mu = np.exp(yellow_fit.params.mu)
    y_alpha = yellow_fit.params.alpha

    # Compound: red Poisson
    red_fit = fit_count_model(
        train_df, "home_reds", "away_reds",
        include_referees=True, min_referee_matches=20,
        alpha_bounds=(1e-12, 1e-12),
    )
    r_mu = np.exp(red_fit.params.mu)

    y_implied_var = y_mu + y_alpha * y_mu ** 2
    r_implied_var = r_mu
    compound_implied = 100 * y_implied_var + 625 * r_implied_var
    compound_mean = 10 * y_mu + 25 * r_mu

    print("\n" + "=" * 70)
    print("  PART 1: Compound Card Model")
    print("=" * 70)
    print(f"  Yellow NB2:  mu(log)={yellow_fit.params.mu:.4f}  "
          f"mu(count)={y_mu:.3f}  alpha={y_alpha:.6f}  gamma={yellow_fit.params.gamma:.4f}")
    print(f"  Red Poisson: mu(log)={red_fit.params.mu:.4f}  "
          f"mu(count)={r_mu:.4f}  alpha={red_fit.params.alpha:.2e}")
    print(f"  Yellow implied var/team:   {y_implied_var:.3f}")
    print(f"  Red implied var/team:      {r_implied_var:.4f}")
    print(f"  Compound implied var(BP):  {compound_implied:.1f}  "
          f"(100*{y_implied_var:.3f} + 625*{r_implied_var:.4f})")
    print(f"  Compound implied mean(BP): {compound_mean:.2f}")
    print(f"  Observed BP var:           {bp_var:.1f}")
    print(f"  Observed BP mean:          {bp_mean:.2f}")
    print(f"  Direct NB2 implied var:    {direct_implied:.1f}  "
          f"(alpha={direct_fit.params.alpha:.4f}, mu={direct_mu:.1f})")
    print(f"  Ratio compound/observed:   {compound_implied / bp_var:.3f}")
    print(f"  Ratio direct/observed:     {direct_implied / bp_var:.3f}")

    # ================================================================
    # PART 2: Corner team effects
    # ================================================================
    corner_train = df[train_mask].dropna(subset=["home_corners", "away_corners"])
    corner_fit = fit_count_model(corner_train, "home_corners", "away_corners")
    atk = corner_fit.params.attack
    dfc = corner_fit.params.defence

    print("\n" + "=" * 70)
    print("  PART 2: Corner Team Effects")
    print("=" * 70)
    print(f"  mu(log)={corner_fit.params.mu:.4f}  gamma={corner_fit.params.gamma:.4f}  "
          f"alpha={corner_fit.params.alpha:.6f}")
    print(f"  attack.std():  {atk.std():.4f}   range: [{atk.min():.4f}, {atk.max():.4f}]  "
          f"ptp: {float(np.ptp(atk)):.4f}")
    print(f"  defence.std(): {dfc.std():.4f}   range: [{dfc.min():.4f}, {dfc.max():.4f}]  "
          f"ptp: {float(np.ptp(dfc)):.4f}")
    print(f"  League-mean corners/side: {np.exp(corner_fit.params.mu):.2f}")
    print(f"  attack effect as % of mu: {atk.std() / abs(corner_fit.params.mu) * 100:.1f}%")
    print(f"  defence effect as % of mu: {dfc.std() / abs(corner_fit.params.mu) * 100:.1f}%")

    # Top/bottom 5 teams by attack
    sorted_atk = sorted(zip(corner_fit.params.teams, atk), key=lambda x: -x[1])
    print("  Top 5 corner attack:")
    for name, val in sorted_atk[:5]:
        print(f"    {name:20s}  atk={val:+.4f}  mult={np.exp(val):.3f}x")
    print("  Bottom 5 corner attack:")
    for name, val in sorted_atk[-5:]:
        print(f"    {name:20s}  atk={val:+.4f}  mult={np.exp(val):.3f}x")

    # ================================================================
    # Run walk-forward backtest
    # ================================================================
    print("\n" + "=" * 70)
    print("  Running walk-forward backtest...")
    print("=" * 70)
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
    print(f"  Done in {elapsed:.0f}s  corners={len(report.corner_predictions)}  "
          f"cards={len(report.card_predictions)}")

    # ================================================================
    # PART 3: Old gate/table mismatch
    # ================================================================
    corner_preds = [CountPrediction(**p) for p in report.corner_predictions]

    all_lines = sorted(set(ou["line"] for ou in corner_preds[0].match_over_under))

    print("\n" + "=" * 70)
    print("  PART 3: Old Gate/Table Mismatch Explained")
    print("=" * 70)
    print(f"  ALL match lines in predictions: {all_lines}")
    print(f"  Gate lines (new, filtered):     {list(CORNER_GATE_LINES)}")
    print()

    all_briers = []
    gate_briers = []
    for line in all_lines:
        model_b, over_rate = per_line_model_brier(corner_preds, line)
        base_b = _compute_baseline_brier(corner_preds, [line])
        marker = " <-- gate" if line in CORNER_GATE_LINES else "     excl"
        print(f"  O/U {line:5.1f}: model={model_b:.6f}  base={base_b:.6f}  "
              f"over_rate={over_rate:.3f}{marker}")
        all_briers.append(model_b)
        if line in CORNER_GATE_LINES:
            gate_briers.append(model_b)

    print(f"\n  Old gate mean Brier (all {len(all_lines)} lines): {np.mean(all_briers):.6f}")
    print(f"  New gate mean Brier ({len(gate_briers)} gate lines):  {np.mean(gate_briers):.6f}")

    # ================================================================
    # PART 4: Gate results on publishable lines
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 4: Gate — Corners (8.5-12.5)")
    print("=" * 70)
    for line in CORNER_GATE_LINES:
        model_b, over_rate = per_line_model_brier(corner_preds, line)
        base_b = _compute_baseline_brier(corner_preds, [line])
        delta_pct = (model_b - base_b) / base_b * 100 if base_b > 0 else 0
        print(f"  O/U {line:5.1f}: model={model_b:.6f}  base={base_b:.6f}  "
              f"delta={delta_pct:+.1f}%  over_rate={over_rate:.3f}")

    corner_gate_model = np.mean(gate_briers)
    corner_gate_base = _compute_baseline_brier(corner_preds, list(CORNER_GATE_LINES))
    print(f"  Gate mean: model={corner_gate_model:.6f}  base={corner_gate_base:.6f}  "
          f"pass={corner_gate_model < corner_gate_base}")

    print("\n" + "=" * 70)
    print("  PART 4: Gate — Booking Points (20.5-50.5)")
    print("=" * 70)
    card_preds = [CountPrediction(**p) for p in report.card_predictions]

    bp_gate_briers = []
    for line in BOOKING_GATE_LINES:
        model_b, over_rate = per_line_model_brier(card_preds, line)
        base_b = _compute_baseline_brier(card_preds, [line])
        delta_pct = (model_b - base_b) / base_b * 100 if base_b > 0 else 0
        print(f"  O/U {line:5.1f}: model={model_b:.6f}  base={base_b:.6f}  "
              f"delta={delta_pct:+.1f}%  over_rate={over_rate:.3f}")
        bp_gate_briers.append(model_b)

    bp_gate_model = np.mean(bp_gate_briers)
    bp_gate_base = _compute_baseline_brier(card_preds, list(BOOKING_GATE_LINES))
    print(f"  Gate mean: model={bp_gate_model:.6f}  base={bp_gate_base:.6f}  "
          f"pass={bp_gate_model < bp_gate_base}")

    # Walk-forward compound stats
    alphas_y = [p.get("alpha_yellow") for p in report.card_predictions
                if p.get("alpha_yellow") is not None]
    mu_r_vals = [p.get("mu_red_home", 0) + p.get("mu_red_away", 0)
                 for p in report.card_predictions if p.get("mu_red_home") is not None]
    print(f"\n  Walk-forward yellow alpha: mean={np.mean(alphas_y):.6f}  "
          f"range=[{np.min(alphas_y):.6f}, {np.max(alphas_y):.6f}]")
    print(f"  Walk-forward red rate (total per match): mean={np.mean(mu_r_vals):.4f}")

    # ================================================================
    # Gate verdicts
    # ================================================================
    print("\n" + "=" * 70)
    print("  GATE VERDICTS")
    print("=" * 70)
    status = "PASS" if report.corner_gate_passed else "FAIL"
    print(f"  Corner gate: {status}")
    for d in report.corner_gate_details:
        s = "PASS" if d.passed else "FAIL"
        print(f"    [{s}] {d.name}: {d.message}")
    status = "PASS" if report.card_gate_passed else "FAIL"
    print(f"  Card gate: {status}")
    for d in report.card_gate_details:
        s = "PASS" if d.passed else "FAIL"
        print(f"    [{s}] {d.name}: {d.message}")


if __name__ == "__main__":
    main()
