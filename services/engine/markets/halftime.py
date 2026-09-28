"""Half-time markets, HT/FT 9-outcome market, and first goal timing.

Pure library — no DB or IO calls. All functions take numpy arrays / ScoreGrids.
"""

from __future__ import annotations

import math

import numpy as np

from services.engine.models.grid import ScoreGrid

# HT O/U lines (reduced set — goals at HT are typically 0-2)
_HT_OU_LINES = [0.5, 1.5, 2.5]

# HT team total lines
_HT_TEAM_LINES = [0.5, 1.5]

# First goal timing cutoff minutes
_FIRST_GOAL_MINUTES = [15, 30, 45]


def _r4(v: float) -> float:
    """Round to 4 decimal places for JSON output."""
    return round(float(v), 4)


def _result(home: int, away: int) -> str:
    """Return 'H', 'D', or 'A' for a scoreline."""
    if home > away:
        return "H"
    elif home == away:
        return "D"
    return "A"


def ht_grid_to_markets(ht_grid: ScoreGrid) -> list[dict]:
    """Derive markets from a half-time score grid.

    Returns a flat list of {market, selection, probability} dicts,
    with all market names prefixed ``ht_``.
    """
    g = ht_grid.grid
    n = g.shape[0]
    markets: list[dict] = []

    # HT 1X2
    home = float(np.tril(g, k=-1).sum())
    draw = float(np.trace(g))
    away = float(np.triu(g, k=1).sum())
    for sel, prob in [("home", home), ("draw", draw), ("away", away)]:
        markets.append({"market": "ht_match_result", "selection": sel, "probability": _r4(prob)})

    # HT Over/Under
    for line in _HT_OU_LINES:
        over = 0.0
        for i in range(n):
            for j in range(n):
                if i + j > line:
                    over += g[i, j]
        mkt = f"ht_over_under_{line}"
        markets.append({"market": mkt, "selection": "over", "probability": _r4(over)})
        markets.append({"market": mkt, "selection": "under", "probability": _r4(1.0 - over)})

    # HT BTTS
    yes = float(g[1:, 1:].sum())
    markets.append({"market": "ht_btts", "selection": "yes", "probability": _r4(yes)})
    markets.append({"market": "ht_btts", "selection": "no", "probability": _r4(1.0 - yes)})

    # HT Correct Score (all cells in the grid)
    for i in range(n):
        for j in range(n):
            markets.append({
                "market": "ht_correct_score",
                "selection": f"{i}-{j}",
                "probability": _r4(float(g[i, j])),
            })

    # HT Team Totals
    home_marginal = g.sum(axis=1)
    away_marginal = g.sum(axis=0)
    for line in _HT_TEAM_LINES:
        threshold = int(line + 0.5)
        h_over = float(home_marginal[threshold:].sum())
        markets.append({
            "market": f"ht_home_total_{line}",
            "selection": "over",
            "probability": _r4(h_over),
        })
        markets.append({
            "market": f"ht_home_total_{line}",
            "selection": "under",
            "probability": _r4(1.0 - h_over),
        })
        a_over = float(away_marginal[threshold:].sum())
        markets.append({
            "market": f"ht_away_total_{line}",
            "selection": "over",
            "probability": _r4(a_over),
        })
        markets.append({
            "market": f"ht_away_total_{line}",
            "selection": "under",
            "probability": _r4(1.0 - a_over),
        })

    return markets


def htft_market(ht_grid: ScoreGrid, sh_grid: ScoreGrid) -> list[dict]:
    """Compute 9-outcome HT/FT market from independent HT and 2H grids.

    Enumerates all (ht_h, ht_a, sh_h, sh_a) combinations, computes
    the full-time score, and accumulates joint HT-result × FT-result
    probabilities.

    Returns 9 {market, selection, probability} dicts.
    Selections: HH, HD, HA, DH, DD, DA, AH, AD, AA.
    """
    ht_g = ht_grid.grid
    sh_g = sh_grid.grid
    ht_n = ht_g.shape[0]
    sh_n = sh_g.shape[0]

    # Accumulate joint probabilities for (HT result, FT result)
    joint = {}
    for ht_r in ("H", "D", "A"):
        for ft_r in ("H", "D", "A"):
            joint[ht_r + ft_r] = 0.0

    for h1 in range(ht_n):
        for a1 in range(ht_n):
            ht_prob = ht_g[h1, a1]
            if ht_prob < 1e-15:
                continue
            ht_res = _result(h1, a1)
            for h2 in range(sh_n):
                for a2 in range(sh_n):
                    sh_prob = sh_g[h2, a2]
                    if sh_prob < 1e-15:
                        continue
                    ft_h = h1 + h2
                    ft_a = a1 + a2
                    ft_res = _result(ft_h, ft_a)
                    joint[ht_res + ft_res] += ht_prob * sh_prob

    outcomes = ["HH", "HD", "HA", "DH", "DD", "DA", "AH", "AD", "AA"]
    return [
        {"market": "htft", "selection": key, "probability": _r4(joint[key])}
        for key in outcomes
    ]


def first_goal_timing(lambda_home: float, lambda_away: float) -> list[dict]:
    """Analytical first-goal timing from Exponential inter-arrival model.

    Goals arrive as a Poisson process with combined rate
    lambda = lambda_home + lambda_away per 90 minutes. The time to first
    goal follows Exponential(lambda).

    Returns {market, selection, probability} dicts for before/after
    each cutoff minute.
    """
    lam = lambda_home + lambda_away
    markets: list[dict] = []

    for minute in _FIRST_GOAL_MINUTES:
        # P(first goal <= minute) = 1 - exp(-lambda * minute/90)
        p_before = 1.0 - math.exp(-lam * minute / 90.0)
        markets.append({
            "market": "first_goal_timing",
            "selection": f"before_{minute}",
            "probability": _r4(p_before),
        })
        markets.append({
            "market": "first_goal_timing",
            "selection": f"after_{minute}",
            "probability": _r4(1.0 - p_before),
        })

    return markets
