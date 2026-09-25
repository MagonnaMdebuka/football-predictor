#!/usr/bin/env python3
"""Diagnostic: total-corner model vs per-team convolution model.

Reports:
- Total model: implied vs observed Var(total)
- Decile calibration: slope
- Gate Brier on [8.5-12.5] vs baseline and simple rate-based model
- Booking point home/away correlation (for future treatment decision)

Usage:
    python scripts/total_corner_diagnostic.py [--csv data/raw/E0.csv ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.engine.models.total_count_fit import fit_total_count_model
from services.engine.models.count_fit import fit_count_model
from services.engine.models.negbin import negbin_pmf
from services.engine.markets.counts import CORNER_GATE_LINES


def load_data(csv_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate CSV files from football-data.co.uk."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # Standardise column names
        col_map = {}
        for col in df.columns:
            if col.upper() in ("HC", "HOMECORNERS"):
                col_map[col] = "home_corners"
            elif col.upper() in ("AC", "AWAYCORNERS"):
                col_map[col] = "away_corners"
            elif col.upper() == "HOMETEAM":
                col_map[col] = "home_team"
            elif col.upper() == "AWAYTEAM":
                col_map[col] = "away_team"
            elif col.upper() == "DATE":
                col_map[col] = "date"
            elif col.upper() in ("HY",):
                col_map[col] = "home_yellows"
            elif col.upper() in ("AY",):
                col_map[col] = "away_yellows"
            elif col.upper() in ("HR",):
                col_map[col] = "home_reds"
            elif col.upper() in ("AR",):
                col_map[col] = "away_reds"
        df = df.rename(columns=col_map)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["home_corners", "away_corners"])
    combined["date"] = pd.to_datetime(combined["date"], dayfirst=True)
    return combined


def compute_variance_diagnostic(df: pd.DataFrame) -> None:
    """Compare observed vs model-implied variance of total corners."""
    total = df["home_corners"] + df["away_corners"]
    observed_var = float(total.var())
    observed_mean = float(total.mean())

    print(f"\n--- Variance Diagnostic ---")
    print(f"N matches:         {len(df)}")
    print(f"Observed mean:     {observed_mean:.2f}")
    print(f"Observed var:      {observed_var:.2f}")

    # Per-team model (convolution) implied variance
    per_team = fit_count_model(df, "home_corners", "away_corners")
    team_to_idx = {t: i for i, t in enumerate(per_team.params.teams)}
    per_team_vars = []
    for _, row in df.iterrows():
        hi = team_to_idx[row["home_team"]]
        ai = team_to_idx[row["away_team"]]
        mu_h = np.exp(
            per_team.params.mu + per_team.params.gamma
            + per_team.params.attack[hi] + per_team.params.defence[ai]
        )
        mu_a = np.exp(
            per_team.params.mu
            + per_team.params.attack[ai] + per_team.params.defence[hi]
        )
        alpha = per_team.params.alpha
        # Var(X+Y) under independence = Var(X) + Var(Y)
        var_h = mu_h + alpha * mu_h**2
        var_a = mu_a + alpha * mu_a**2
        per_team_vars.append(var_h + var_a)
    conv_var = float(np.mean(per_team_vars))
    print(f"Convolution implied var: {conv_var:.2f} (ratio: {conv_var / observed_var:.3f})")

    # Total model implied variance
    total_model = fit_total_count_model(df, "home_corners", "away_corners")
    total_team_idx = {t: i for i, t in enumerate(total_model.params.teams)}
    total_vars = []
    for _, row in df.iterrows():
        hi = total_team_idx[row["home_team"]]
        ai = total_team_idx[row["away_team"]]
        mu_t = np.exp(
            total_model.params.mu
            + total_model.params.home_effect[hi]
            + total_model.params.away_effect[ai]
        )
        alpha_t = total_model.params.alpha
        var_t = mu_t + alpha_t * mu_t**2
        total_vars.append(var_t)
    total_var = float(np.mean(total_vars))
    print(f"Total model implied var: {total_var:.2f} (ratio: {total_var / observed_var:.3f})")


def compute_brier_diagnostic(df: pd.DataFrame) -> None:
    """Compare Brier scores on gate lines for total model vs convolution."""
    total = (df["home_corners"] + df["away_corners"]).values

    # Fit both models
    per_team = fit_count_model(df, "home_corners", "away_corners")
    total_model = fit_total_count_model(df, "home_corners", "away_corners")

    team_to_idx = {t: i for i, t in enumerate(per_team.params.teams)}
    total_team_idx = {t: i for i, t in enumerate(total_model.params.teams)}

    # Base-rate: P(over L) = fraction of matches with total > L
    base_briers = {}
    conv_briers = {}
    total_briers = {}

    for line in CORNER_GATE_LINES:
        base_over = float(np.mean(total > line))
        base_brier_sum = 0.0
        conv_brier_sum = 0.0
        total_brier_sum = 0.0
        n = len(df)

        for idx, (_, row) in enumerate(df.iterrows()):
            actual = 1.0 if total[idx] > line else 0.0

            # Base rate
            base_brier_sum += (base_over - actual) ** 2

            hi = team_to_idx[row["home_team"]]
            ai = team_to_idx[row["away_team"]]

            # Convolution model P(over)
            mu_h = np.exp(
                per_team.params.mu + per_team.params.gamma
                + per_team.params.attack[hi] + per_team.params.defence[ai]
            )
            mu_a = np.exp(
                per_team.params.mu
                + per_team.params.attack[ai] + per_team.params.defence[hi]
            )
            pmf_h = negbin_pmf(np.arange(81), mu_h, per_team.params.alpha)
            pmf_a = negbin_pmf(np.arange(81), mu_a, per_team.params.alpha)
            pmf_conv = np.convolve(pmf_h, pmf_a)
            cdf_conv = np.cumsum(pmf_conv)
            k = int(line)
            p_over_conv = 1.0 - float(cdf_conv[k]) if k < len(cdf_conv) else 0.0
            conv_brier_sum += (p_over_conv - actual) ** 2

            # Total model P(over)
            thi = total_team_idx[row["home_team"]]
            tai = total_team_idx[row["away_team"]]
            mu_t = np.exp(
                total_model.params.mu
                + total_model.params.home_effect[thi]
                + total_model.params.away_effect[tai]
            )
            pmf_t = negbin_pmf(np.arange(81), mu_t, total_model.params.alpha)
            cdf_t = np.cumsum(pmf_t)
            p_over_total = 1.0 - float(cdf_t[k]) if k < len(cdf_t) else 0.0
            total_brier_sum += (p_over_total - actual) ** 2

        base_briers[line] = base_brier_sum / n
        conv_briers[line] = conv_brier_sum / n
        total_briers[line] = total_brier_sum / n

    mean_base = np.mean(list(base_briers.values()))
    mean_conv = np.mean(list(conv_briers.values()))
    mean_total = np.mean(list(total_briers.values()))

    print(f"\n--- Brier Score Diagnostic (gate lines {CORNER_GATE_LINES}) ---")
    print(f"{'Line':>6}  {'Base':>8}  {'Convol':>8}  {'Total':>8}")
    for line in CORNER_GATE_LINES:
        print(f"{line:>6.1f}  {base_briers[line]:>8.4f}  "
              f"{conv_briers[line]:>8.4f}  {total_briers[line]:>8.4f}")
    print(f"{'Mean':>6}  {mean_base:>8.4f}  {mean_conv:>8.4f}  {mean_total:>8.4f}")

    conv_diff = (mean_conv - mean_base) / mean_base * 100
    total_diff = (mean_total - mean_base) / mean_base * 100
    print(f"\nConvolution vs base: {conv_diff:+.2f}%")
    print(f"Total model vs base: {total_diff:+.2f}%")


def compute_booking_correlation(df: pd.DataFrame) -> None:
    """Check home/away booking point correlation."""
    print(f"\n--- Booking Point Correlation ---")

    # Compute booking points if not present
    bp_cols = ["home_yellows", "away_yellows", "home_reds", "away_reds"]
    if all(c in df.columns for c in bp_cols):
        bp_df = df.dropna(subset=bp_cols)
        if len(bp_df) > 10:
            home_bp = 10 * bp_df["home_yellows"] + 25 * bp_df["home_reds"]
            away_bp = 10 * bp_df["away_yellows"] + 25 * bp_df["away_reds"]
            r, p = pearsonr(home_bp, away_bp)
            print(f"N matches: {len(bp_df)}")
            print(f"Pearson r: {r:.3f}, p = {p:.6f}")
            if abs(r) > 0.15 and p < 0.05:
                print("=> Material correlation — consider total model for booking points")
            else:
                print("=> Weak/insignificant correlation — convolution acceptable")
        else:
            print("Insufficient card data")
    else:
        print("Card columns not found in data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Total corner model diagnostic")
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="CSV file paths (football-data.co.uk format)",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    print(f"Loaded {len(df)} matches from {len(args.csv)} file(s)")

    compute_variance_diagnostic(df)
    compute_brier_diagnostic(df)
    compute_booking_correlation(df)


if __name__ == "__main__":
    main()
