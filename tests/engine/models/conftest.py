"""Shared fixtures for Dixon-Coles model tests.

Provides:
- 4-team synthetic dataset (96 matches with known parameters)
- Sample CSV as a DataFrame (20-row real data from tests/fixtures/sample_e0.csv)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Known "true" parameters for the synthetic dataset
# ---------------------------------------------------------------------------
TRUE_TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]
TRUE_MU = 0.25       # home advantage
TRUE_GAMMA = 0.20    # overall scoring rate
TRUE_RHO = -0.10     # dependence parameter
TRUE_ATTACK = np.array([0.30, 0.10, -0.15, -0.25])   # sums to 0
TRUE_DEFENCE = np.array([-0.20, 0.05, 0.10, 0.05])   # sums to 0


@pytest.fixture
def true_teams() -> list[str]:
    return list(TRUE_TEAMS)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Generate 96 matches (4 teams × 12 pairings × 8 rounds) with Poisson goals.

    Goals are drawn from Poisson with the known true parameters, giving the
    optimiser a realistic dataset to recover them from.
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    base_date = np.datetime64("2024-01-01")

    round_num = 0
    for _repeat in range(8):
        for i, home in enumerate(TRUE_TEAMS):
            for j, away in enumerate(TRUE_TEAMS):
                if i == j:
                    continue
                lam_h = np.exp(
                    TRUE_MU + TRUE_GAMMA + TRUE_ATTACK[i] + TRUE_DEFENCE[j]
                )
                lam_a = np.exp(
                    TRUE_GAMMA + TRUE_ATTACK[j] + TRUE_DEFENCE[i]
                )
                rows.append({
                    "date": base_date + np.timedelta64(round_num * 7, "D"),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": int(rng.poisson(lam_h)),
                    "away_goals": int(rng.poisson(lam_a)),
                })
                round_num += 1

    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv_df() -> pd.DataFrame:
    """Load the 20-row sample CSV as a DataFrame with columns the model expects."""
    csv_path = Path(__file__).resolve().parents[2] / "fixtures" / "sample_e0.csv"
    raw = pd.read_csv(csv_path)
    return pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True),
        "home_team": raw["HomeTeam"],
        "away_team": raw["AwayTeam"],
        "home_goals": raw["FTHG"],
        "away_goals": raw["FTAG"],
    })
