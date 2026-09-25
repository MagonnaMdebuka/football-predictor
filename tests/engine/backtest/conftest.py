"""Shared fixtures for backtest tests.

Provides:
- 6-team synthetic dataset spanning seasons 2019-20 to 2025-26
- Sample source_rows dicts with odds data
- Default BacktestConfig
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from services.engine.backtest.types import BacktestConfig, BookmakerOddsCols

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
TRUE_MU = 0.20      # league scoring-rate intercept
TRUE_GAMMA = 0.25   # home-advantage parameter
TRUE_RHO = -0.08
TRUE_ATTACK = np.array([0.30, 0.15, 0.05, -0.10, -0.15, -0.25])
TRUE_DEFENCE = np.array([-0.20, -0.05, 0.05, 0.10, 0.05, 0.05])

# Count model true parameters for synthetic data
CORNER_MU = 2.20        # log-scale intercept for corners (~9 per side)
CORNER_GAMMA = 0.10     # home advantage for corners
CORNER_ALPHA = 0.15     # NB2 overdispersion
# Compound card model: yellows from NB2, reds from Poisson, bp = 10*Y + 25*R
YELLOW_MU = 1.10        # log-scale intercept for yellows (~3.0 per side)
YELLOW_ALPHA = 0.15     # NB2 overdispersion for yellows
RED_MU = -2.30          # log-scale intercept for reds (~0.1 per side)
CARD_GAMMA = 0.05       # home advantage for cards (shared by yellow/red)
REFEREES = ["Smith", "Jones", "Taylor", "Brown", "Wilson"]
REF_EFFECT = np.array([0.15, 0.05, -0.05, -0.10, -0.05])  # sums to 0

# Season date ranges (August to May, matching football-data.co.uk)
SEASON_RANGES = {
    "2019-20": (date(2019, 8, 10), date(2020, 5, 17)),
    "2020-21": (date(2020, 8, 12), date(2021, 5, 23)),
    "2021-22": (date(2021, 8, 14), date(2022, 5, 22)),
    "2022-23": (date(2022, 8, 6), date(2023, 5, 28)),
    "2023-24": (date(2023, 8, 12), date(2024, 5, 19)),
    "2024-25": (date(2024, 8, 17), date(2025, 5, 25)),
    "2025-26": (date(2025, 8, 16), date(2026, 5, 24)),
}


def _generate_season_matches(
    season: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate round-robin matches for one season."""
    start, end = SEASON_RANGES[season]
    # Each pair plays twice (home and away) = n*(n-1) matches per season
    fixtures = []
    for i, home in enumerate(TEAMS):
        for j, away in enumerate(TEAMS):
            if i == j:
                continue
            fixtures.append((home, away, i, j))

    n_matches = len(fixtures)
    total_days = (end - start).days
    match_dates = [
        start + timedelta(days=int(k * total_days / n_matches))
        for k in range(n_matches)
    ]

    rows = []
    for idx, (home, away, hi, ai) in enumerate(fixtures):
        lam_h = np.exp(TRUE_MU + TRUE_GAMMA + TRUE_ATTACK[hi] + TRUE_DEFENCE[ai])
        lam_a = np.exp(TRUE_MU + TRUE_ATTACK[ai] + TRUE_DEFENCE[hi])
        hg = int(rng.poisson(lam_h))
        ag = int(rng.poisson(lam_a))

        # Generate synthetic odds (slight noise around true probabilities)
        p_home = max(0.15, min(0.70, 0.45 + 0.1 * (TRUE_ATTACK[hi] - TRUE_ATTACK[ai])))
        p_draw = max(0.15, min(0.40, 0.28))
        p_away = 1.0 - p_home - p_draw
        p_away = max(0.10, p_away)
        total_p = p_home + p_draw + p_away
        # Pinnacle-style odds with ~2.5% overround
        overround = 1.025
        odds_h = overround / (p_home / total_p)
        odds_d = overround / (p_draw / total_p)
        odds_a = overround / (p_away / total_p)
        # Market average with ~5% overround
        avg_overround = 1.05
        avg_odds_h = avg_overround / (p_home / total_p)
        avg_odds_d = avg_overround / (p_draw / total_p)
        avg_odds_a = avg_overround / (p_away / total_p)

        raw_json = json.dumps({
            "PSCH": round(odds_h, 2),
            "PSCD": round(odds_d, 2),
            "PSCA": round(odds_a, 2),
            "AvgCH": round(avg_odds_h, 2),
            "AvgCD": round(avg_odds_d, 2),
            "AvgCA": round(avg_odds_a, 2),
        })

        if hg > ag:
            ftr = "H"
        elif hg == ag:
            ftr = "D"
        else:
            ftr = "A"

        # Generate synthetic corner counts (NB2-distributed)
        from scipy.stats import nbinom as nbinom_dist
        corner_mu_h = np.exp(CORNER_MU + CORNER_GAMMA)
        corner_mu_a = np.exp(CORNER_MU)
        cn_h = 1.0 / CORNER_ALPHA
        cp_h = cn_h / (cn_h + corner_mu_h)
        cn_a = 1.0 / CORNER_ALPHA
        cp_a = cn_a / (cn_a + corner_mu_a)
        hc = int(nbinom_dist.rvs(cn_h, cp_h, random_state=rng))
        ac = int(nbinom_dist.rvs(cn_a, cp_a, random_state=rng))

        # Generate synthetic card counts: yellows (NB2), reds (Poisson), bp = 10*Y + 25*R
        ref_idx = rng.integers(0, len(REFEREES))
        ref_name = REFEREES[ref_idx]
        ref_eff = REF_EFFECT[ref_idx]
        # Yellows from NB2
        yellow_mu_h = np.exp(YELLOW_MU + CARD_GAMMA + ref_eff)
        yellow_mu_a = np.exp(YELLOW_MU + ref_eff)
        yn_h = 1.0 / YELLOW_ALPHA
        yp_h = yn_h / (yn_h + yellow_mu_h)
        yn_a = 1.0 / YELLOW_ALPHA
        yp_a = yn_a / (yn_a + yellow_mu_a)
        h_yellows = int(nbinom_dist.rvs(yn_h, yp_h, random_state=rng))
        a_yellows = int(nbinom_dist.rvs(yn_a, yp_a, random_state=rng))
        # Reds from Poisson (alpha~0)
        red_mu_h = np.exp(RED_MU + CARD_GAMMA + ref_eff)
        red_mu_a = np.exp(RED_MU + ref_eff)
        h_reds = int(rng.poisson(red_mu_h))
        a_reds = int(rng.poisson(red_mu_a))
        h_bp = 10 * h_yellows + 25 * h_reds
        a_bp = 10 * a_yellows + 25 * a_reds

        rows.append({
            "date": match_dates[idx],
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "ftr": ftr,
            "source_row_raw": raw_json,
            "home_corners": hc,
            "away_corners": ac,
            "home_yellows": h_yellows,
            "away_yellows": a_yellows,
            "home_reds": h_reds,
            "away_reds": a_reds,
            "home_booking_points": h_bp,
            "away_booking_points": a_bp,
            "referee": ref_name,
        })

    return rows


@pytest.fixture
def multi_season_df() -> pd.DataFrame:
    """Generate a multi-season synthetic DataFrame (6 teams, 2019-20 to 2025-26).

    Each season has 30 matches (6 teams, each pair plays once in each direction).
    A promoted team "Golf" appears only in held-out seasons (2024-25, 2025-26)
    to exercise the fallback-for-missing-team logic.
    Total: 210 regular + 2 promoted = 212 matches across 7 seasons.
    """
    rng = np.random.default_rng(42)
    all_rows = []
    for season in SEASON_RANGES:
        all_rows.extend(_generate_season_matches(season, rng))

    # Add a promoted team "Golf" in each held-out season. Golf has never
    # appeared in training, so the harness must use fallback strengths.
    promoted_matches = [
        {
            "date": date(2024, 8, 17),
            "season": "2024-25",
            "home_team": "Golf",
            "away_team": "Alpha",
            "home_goals": 1,
            "away_goals": 2,
            "ftr": "A",
            "source_row_raw": json.dumps({
                "PSCH": 3.50, "PSCD": 3.20, "PSCA": 2.10,
                "AvgCH": 3.60, "AvgCD": 3.30, "AvgCA": 2.15,
            }),
            "home_corners": 4,
            "away_corners": 7,
            "home_yellows": 2,
            "away_yellows": 1,
            "home_reds": 0,
            "away_reds": 1,
            "home_booking_points": 20,
            "away_booking_points": 35,
            "referee": "Smith",
        },
        {
            "date": date(2025, 8, 16),
            "season": "2025-26",
            "home_team": "Alpha",
            "away_team": "Golf",
            "home_goals": 3,
            "away_goals": 0,
            "ftr": "H",
            "source_row_raw": json.dumps({
                "PSCH": 1.80, "PSCD": 3.40, "PSCA": 4.50,
                "AvgCH": 1.85, "AvgCD": 3.50, "AvgCA": 4.60,
            }),
            "home_corners": 8,
            "away_corners": 3,
            "home_yellows": 2,
            "away_yellows": 3,
            "home_reds": 0,
            "away_reds": 0,
            "home_booking_points": 20,
            "away_booking_points": 30,
            "referee": "Jones",
        },
    ]
    all_rows.extend(promoted_matches)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def default_config() -> BacktestConfig:
    """Default backtest configuration for tests."""
    return BacktestConfig(
        held_out_seasons=("2024-25", "2025-26"),
        training_start_season="2019-20",
        xi=0.0065,
        refit_step="per_date",
        weekly_refit_day=0,  # Monday
        rho_bounds=(-0.5, 0.5),
        max_goals=11,
        seed=42,
        bookmaker_odds_cols=BookmakerOddsCols(),
    )


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Small 4-team DataFrame for quick metric tests."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"] * 6),
        "home_team": ["A", "B", "C", "D", "A", "B"],
        "away_team": ["B", "C", "D", "A", "C", "D"],
        "home_goals": [2, 1, 0, 3, 1, 2],
        "away_goals": [1, 1, 2, 0, 0, 1],
        "ftr": ["H", "D", "A", "H", "H", "H"],
    })
