"""Bootstrap calibration data from backtest JSON.

Loads a walk-forward backtest report and derives calibration maps
for each market+selection from the 760-match out-of-sample predictions.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import poisson

from services.engine.calibration.isotonic import fit_calibration_bins


def load_backtest_predictions(json_path: str) -> list[dict]:
    """Load backtest JSON and extract per-match predictions.

    Returns list of dicts with:
        result, home_goals, away_goals,
        model_home, model_draw, model_away,
        lambda_home, lambda_away,
        bookmaker_home, bookmaker_draw, bookmaker_away
    """
    with open(json_path) as f:
        data = json.load(f)

    predictions = []
    for p in data["predictions"]:
        predictions.append({
            "result": p["result"],
            "home_goals": p["home_goals"],
            "away_goals": p["away_goals"],
            "model_home": p["model_home"],
            "model_draw": p["model_draw"],
            "model_away": p["model_away"],
            "lambda_home": p["lambda_home"],
            "lambda_away": p["lambda_away"],
            "bookmaker_home": p.get("bookmaker_home"),
            "bookmaker_draw": p.get("bookmaker_draw"),
            "bookmaker_away": p.get("bookmaker_away"),
        })

    return predictions


def _poisson_grid(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    """Build an independent Poisson probability grid from lambdas."""
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lam_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lam_away)
    return np.outer(home_pmf, away_pmf)


def derive_market_outcomes(predictions: list[dict]) -> dict[str, list[dict]]:
    """From backtest predictions, derive observed outcomes for each market.

    For 1X2: uses model_home/draw/away directly.
    For O/U and BTTS: reconstructs from lambda_home + lambda_away
    using independent Poisson (no rho correction).

    Returns {market_selection: [{predicted_prob, observed}]} for binary markets,
    or {market: [{probs, outcome_idx}]} for 1X2.
    """
    outcomes: dict[str, list[dict]] = {
        "match_result": [],
        "match_result_home": [],
        "match_result_draw": [],
        "match_result_away": [],
        "over_under_2.5_over": [],
        "over_under_2.5_under": [],
        "btts_yes": [],
        "btts_no": [],
        "over_under_1.5_over": [],
        "over_under_1.5_under": [],
        "over_under_3.5_over": [],
        "over_under_3.5_under": [],
    }

    for p in predictions:
        hg = p["home_goals"]
        ag = p["away_goals"]
        result = p["result"]

        # 1X2 outcome index: H=0, D=1, A=2
        if result == "H":
            outcome_idx = 0
        elif result == "D":
            outcome_idx = 1
        else:
            outcome_idx = 2

        # 1X2 (multi-outcome)
        outcomes["match_result"].append({
            "probs": [p["model_home"], p["model_draw"], p["model_away"]],
            "outcome_idx": outcome_idx,
        })

        # 1X2 binary per selection
        outcomes["match_result_home"].append({
            "predicted_prob": p["model_home"],
            "observed": 1 if result == "H" else 0,
        })
        outcomes["match_result_draw"].append({
            "predicted_prob": p["model_draw"],
            "observed": 1 if result == "D" else 0,
        })
        outcomes["match_result_away"].append({
            "predicted_prob": p["model_away"],
            "observed": 1 if result == "A" else 0,
        })

        # Reconstruct grid from lambdas for goal-derived markets
        grid = _poisson_grid(p["lambda_home"], p["lambda_away"])
        total_goals = hg + ag

        # Over/Under 2.5
        prob_over_25 = float(1.0 - sum(
            grid[i, j] for i in range(grid.shape[0])
            for j in range(grid.shape[1]) if i + j <= 2
        ))
        outcomes["over_under_2.5_over"].append({
            "predicted_prob": prob_over_25,
            "observed": 1 if total_goals > 2 else 0,
        })
        outcomes["over_under_2.5_under"].append({
            "predicted_prob": 1.0 - prob_over_25,
            "observed": 1 if total_goals <= 2 else 0,
        })

        # Over/Under 1.5
        prob_over_15 = float(1.0 - sum(
            grid[i, j] for i in range(grid.shape[0])
            for j in range(grid.shape[1]) if i + j <= 1
        ))
        outcomes["over_under_1.5_over"].append({
            "predicted_prob": prob_over_15,
            "observed": 1 if total_goals > 1 else 0,
        })
        outcomes["over_under_1.5_under"].append({
            "predicted_prob": 1.0 - prob_over_15,
            "observed": 1 if total_goals <= 1 else 0,
        })

        # Over/Under 3.5
        prob_over_35 = float(1.0 - sum(
            grid[i, j] for i in range(grid.shape[0])
            for j in range(grid.shape[1]) if i + j <= 3
        ))
        outcomes["over_under_3.5_over"].append({
            "predicted_prob": prob_over_35,
            "observed": 1 if total_goals > 3 else 0,
        })
        outcomes["over_under_3.5_under"].append({
            "predicted_prob": 1.0 - prob_over_35,
            "observed": 1 if total_goals <= 3 else 0,
        })

        # BTTS
        prob_btts = float(1.0 - sum(
            grid[i, j] for i in range(grid.shape[0])
            for j in range(grid.shape[1]) if i == 0 or j == 0
        ))
        outcomes["btts_yes"].append({
            "predicted_prob": prob_btts,
            "observed": 1 if hg > 0 and ag > 0 else 0,
        })
        outcomes["btts_no"].append({
            "predicted_prob": 1.0 - prob_btts,
            "observed": 1 if hg == 0 or ag == 0 else 0,
        })

    return outcomes


def lookup_calibrated_prob(
    bins: list[dict],
    raw_prob: float,
) -> float:
    """Look up the calibrated probability for a raw prediction.

    Finds the bin whose [bin_lower, bin_upper) range contains raw_prob
    and returns the isotonic-corrected observed_frequency.
    """
    for i, b in enumerate(bins):
        lo, hi = b["bin_lower"], b["bin_upper"]
        if i < len(bins) - 1:
            if lo <= raw_prob < hi:
                return b["observed_frequency"]
        else:
            # Last bin includes right edge
            if lo <= raw_prob <= hi:
                return b["observed_frequency"]
    # Fallback: return raw prob if outside all bins
    return raw_prob


def calibrate_and_renormalise(
    calibration_maps: dict[str, list[dict]],
    raw_probs: dict[str, float],
    group: list[str],
) -> dict[str, float]:
    """Look up calibrated probs for each selection and renormalise to sum to 1.

    Parameters
    ----------
    calibration_maps : calibration bins keyed by market_selection
    raw_probs : raw model probabilities keyed by market_selection
    group : list of market_selection keys that must sum to 1

    Returns
    -------
    Dict of {market_selection: calibrated_renormalised_prob}.
    """
    calibrated = {}
    for key in group:
        if key in calibration_maps and key in raw_probs:
            calibrated[key] = lookup_calibrated_prob(
                calibration_maps[key], raw_probs[key]
            )
        elif key in raw_probs:
            calibrated[key] = raw_probs[key]

    total = sum(calibrated.values())
    if total > 0:
        calibrated = {k: v / total for k, v in calibrated.items()}

    return calibrated


# Groups of markets that must sum to 1 after calibration
RENORM_GROUPS: list[list[str]] = [
    ["match_result_home", "match_result_draw", "match_result_away"],
    ["over_under_2.5_over", "over_under_2.5_under"],
    ["over_under_1.5_over", "over_under_1.5_under"],
    ["over_under_3.5_over", "over_under_3.5_under"],
    ["btts_yes", "btts_no"],
]


def build_calibration_maps_from_backtest(
    json_path: str,
    n_bins: int = 10,
) -> dict[str, list[dict]]:
    """Full pipeline: load backtest -> derive outcomes -> fit isotonic bins.

    Calibration bins are fitted independently per selection. Renormalisation
    to ensure grouped markets (1X2 triplet, complementary pairs) sum to 1
    is applied per-prediction at lookup time via ``calibrate_and_renormalise``.

    Returns {market_selection: [bin_dicts]}.
    """
    predictions = load_backtest_predictions(json_path)
    market_outcomes = derive_market_outcomes(predictions)

    calibration_maps: dict[str, list[dict]] = {}

    for market_key, obs_list in market_outcomes.items():
        if market_key == "match_result":
            # Skip the multi-outcome entry; calibration is per-selection
            continue

        if len(obs_list) == 0:
            continue

        predicted = np.array([o["predicted_prob"] for o in obs_list])
        observed = np.array([o["observed"] for o in obs_list])
        bins = fit_calibration_bins(predicted, observed, n_bins=n_bins)
        calibration_maps[market_key] = bins

    return calibration_maps
