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
TRUE_MU = 0.20       # league scoring-rate intercept
TRUE_GAMMA = 0.25    # home-advantage parameter
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
                    TRUE_MU + TRUE_ATTACK[j] + TRUE_DEFENCE[i]
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


# ---------------------------------------------------------------------------
# Count model fixtures (corners and cards)
# ---------------------------------------------------------------------------
COUNT_TEAMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
COUNT_MU_CORNERS = 2.20     # log-scale: exp(2.2) ~ 9.0 corners per side
COUNT_GAMMA_CORNERS = 0.10  # slight home advantage for corners
COUNT_ALPHA_CORNERS = 0.15  # moderate overdispersion
COUNT_ATTACK_CORNERS = np.array([0.15, 0.10, 0.05, -0.05, -0.10, -0.15])
COUNT_DEFENCE_CORNERS = np.array([-0.10, -0.05, 0.02, 0.03, 0.05, 0.05])

COUNT_MU_CARDS = 3.20       # log-scale: exp(3.2) ~ 24.5 booking points per side
COUNT_GAMMA_CARDS = 0.05    # slight home advantage for cards
COUNT_ALPHA_CARDS = 0.25    # more overdispersion for booking points
COUNT_ATTACK_CARDS = np.array([0.10, 0.05, 0.02, -0.02, -0.05, -0.10])
COUNT_DEFENCE_CARDS = np.array([-0.08, -0.03, 0.01, 0.02, 0.04, 0.04])

REFEREES = ["Smith", "Jones", "Taylor", "Brown", "Wilson"]
REF_EFFECT = np.array([0.15, 0.05, -0.05, -0.10, -0.05])  # sums to 0


@pytest.fixture
def synthetic_corners_df() -> pd.DataFrame:
    """Generate 180 matches (6 teams, home+away) with NB2-distributed corner counts.

    Uses the count model parameters above so the optimiser can recover them.
    """
    from scipy.stats import nbinom

    rng = np.random.default_rng(42)
    rows: list[dict] = []
    base_date = np.datetime64("2024-01-01")
    match_num = 0

    for _repeat in range(6):
        for i, home in enumerate(COUNT_TEAMS):
            for j, away in enumerate(COUNT_TEAMS):
                if i == j:
                    continue
                mu_h = np.exp(
                    COUNT_MU_CORNERS + COUNT_GAMMA_CORNERS
                    + COUNT_ATTACK_CORNERS[i] + COUNT_DEFENCE_CORNERS[j]
                )
                mu_a = np.exp(
                    COUNT_MU_CORNERS
                    + COUNT_ATTACK_CORNERS[j] + COUNT_DEFENCE_CORNERS[i]
                )
                alpha = COUNT_ALPHA_CORNERS
                # NB2 -> scipy nbinom params
                n_h = 1.0 / alpha
                p_h = n_h / (n_h + mu_h)
                n_a = 1.0 / alpha
                p_a = n_a / (n_a + mu_a)

                rows.append({
                    "date": base_date + np.timedelta64(match_num * 3, "D"),
                    "home_team": home,
                    "away_team": away,
                    "home_corners": int(nbinom.rvs(n_h, p_h, random_state=rng)),
                    "away_corners": int(nbinom.rvs(n_a, p_a, random_state=rng)),
                })
                match_num += 1

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_cards_df() -> pd.DataFrame:
    """Generate 180 matches with NB2-distributed booking point counts + referee.

    Each match gets a random referee from the REFEREES list, with a referee
    effect added to the expected counts.
    """
    from scipy.stats import nbinom

    rng = np.random.default_rng(99)
    rows: list[dict] = []
    base_date = np.datetime64("2024-01-01")
    match_num = 0

    for _repeat in range(6):
        for i, home in enumerate(COUNT_TEAMS):
            for j, away in enumerate(COUNT_TEAMS):
                if i == j:
                    continue
                ref_idx = rng.integers(0, len(REFEREES))
                ref_name = REFEREES[ref_idx]
                ref_eff = REF_EFFECT[ref_idx]

                mu_h = np.exp(
                    COUNT_MU_CARDS + COUNT_GAMMA_CARDS
                    + COUNT_ATTACK_CARDS[i] + COUNT_DEFENCE_CARDS[j]
                    + ref_eff
                )
                mu_a = np.exp(
                    COUNT_MU_CARDS
                    + COUNT_ATTACK_CARDS[j] + COUNT_DEFENCE_CARDS[i]
                    + ref_eff
                )
                alpha = COUNT_ALPHA_CARDS
                n_h = 1.0 / alpha
                p_h = n_h / (n_h + mu_h)
                n_a = 1.0 / alpha
                p_a = n_a / (n_a + mu_a)

                rows.append({
                    "date": base_date + np.timedelta64(match_num * 3, "D"),
                    "home_team": home,
                    "away_team": away,
                    "home_booking_points": int(nbinom.rvs(n_h, p_h, random_state=rng)),
                    "away_booking_points": int(nbinom.rvs(n_a, p_a, random_state=rng)),
                    "referee": ref_name,
                })
                match_num += 1

    return pd.DataFrame(rows)
