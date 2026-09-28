"""Half-time backtest harness — fit HT and 2H Dixon-Coles models.

Fits separate Dixon-Coles models on half-time goals and second-half goals
(FT − HT), then generates HalfTimePrediction records including HT/FT
9-outcome market probabilities and first-goal timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from services.engine.backtest.types import HalfTimePrediction
from services.engine.markets.halftime import htft_market
from services.engine.models.fit import fit_dixon_coles
from services.engine.models.grid import build_grid
from services.engine.models.params import DixonColesParams, pack

HT_MAX_GOALS = 7


def _augment_params(
    params: DixonColesParams,
    teams_needed: list[str],
) -> DixonColesParams:
    """Add missing teams at league-average strength (attack=0, defence=0)."""
    missing = [t for t in teams_needed if t not in params.teams]
    if not missing:
        return params

    new_teams = list(params.teams) + missing
    new_attack = np.concatenate([params.attack, np.zeros(len(missing))])
    new_defence = np.concatenate([params.defence, np.zeros(len(missing))])

    return DixonColesParams(
        teams=new_teams,
        mu=params.mu,
        attack=new_attack,
        defence=new_defence,
        gamma=params.gamma,
        rho=params.rho,
    )


def _result(home: int, away: int) -> str:
    if home > away:
        return "H"
    elif home == away:
        return "D"
    return "A"


def run_halftime_predictions(
    training_df: pd.DataFrame,
    prediction_matches: pd.DataFrame,
    weights: NDArray[np.float64] | None = None,
    prev_ht_packed: NDArray[np.float64] | None = None,
    prev_ht_teams: list[str] | None = None,
    prev_sh_packed: NDArray[np.float64] | None = None,
    prev_sh_teams: list[str] | None = None,
) -> tuple[
    list[HalfTimePrediction],
    NDArray[np.float64], list[str],
    NDArray[np.float64], list[str],
]:
    """Fit HT and 2H Dixon-Coles models and predict for one refit date.

    Returns:
        (predictions, ht_packed, ht_teams, sh_packed, sh_teams)
        for warm-start chaining.
    """
    empty: tuple[
        list[HalfTimePrediction],
        NDArray[np.float64], list[str],
        NDArray[np.float64], list[str],
    ] = ([], np.array([]), [], np.array([]), [])

    # Require HT goal columns
    if "ht_home_goals" not in training_df.columns or "ht_away_goals" not in training_df.columns:
        return empty

    # Drop rows with missing HT goals
    ht_clean = training_df.dropna(subset=["ht_home_goals", "ht_away_goals"]).copy()
    if len(ht_clean) < 10:
        return empty

    if len(prediction_matches) == 0:
        return empty

    # Team list for warm-start comparison
    teams = sorted(
        set(ht_clean["home_team"].unique()) | set(ht_clean["away_team"].unique())
    )

    # Prepare HT training data: rename HT goals to home_goals/away_goals
    ht_df = ht_clean[["home_team", "away_team", "date"]].copy()
    ht_df["home_goals"] = ht_clean["ht_home_goals"].astype(int)
    ht_df["away_goals"] = ht_clean["ht_away_goals"].astype(int)

    # Prepare 2H training data: second-half goals = FT - HT
    sh_df = ht_clean[["home_team", "away_team", "date"]].copy()
    sh_df["home_goals"] = (ht_clean["home_goals"] - ht_clean["ht_home_goals"]).astype(int)
    sh_df["away_goals"] = (ht_clean["away_goals"] - ht_clean["ht_away_goals"]).astype(int)

    # Warm-start: use previous params if team set unchanged
    ht_x0 = None
    if prev_ht_packed is not None and prev_ht_teams is not None and teams == prev_ht_teams:
        ht_x0 = prev_ht_packed

    sh_x0 = None
    if prev_sh_packed is not None and prev_sh_teams is not None and teams == prev_sh_teams:
        sh_x0 = prev_sh_packed

    # Fit HT Dixon-Coles
    ht_fit = fit_dixon_coles(ht_df, weights=weights, x0=ht_x0)
    ht_packed = pack(ht_fit.params)

    # Fit 2H Dixon-Coles
    sh_fit = fit_dixon_coles(sh_df, weights=weights, x0=sh_x0)
    sh_packed = pack(sh_fit.params)

    # Generate predictions
    predictions: list[HalfTimePrediction] = []

    for _, match in prediction_matches.iterrows():
        ht = match["home_team"]
        at = match["away_team"]

        ht_params = _augment_params(ht_fit.params, [ht, at])
        sh_params = _augment_params(sh_fit.params, [ht, at])

        ht_grid = build_grid(ht_params, ht, at, max_goals=HT_MAX_GOALS)
        sh_grid = build_grid(sh_params, ht, at, max_goals=HT_MAX_GOALS)

        # HT/FT market
        htft = htft_market(ht_grid, sh_grid)
        htft_probs = {m["selection"]: m["probability"] for m in htft}

        # Actual goals
        ht_hg = int(match["ht_home_goals"]) if pd.notna(match.get("ht_home_goals")) else 0
        ht_ag = int(match["ht_away_goals"]) if pd.notna(match.get("ht_away_goals")) else 0
        ft_hg = int(match["home_goals"])
        ft_ag = int(match["away_goals"])
        sh_hg = ft_hg - ht_hg
        sh_ag = ft_ag - ht_ag

        pred = HalfTimePrediction(
            date=pd.Timestamp(match["date"]).strftime("%Y-%m-%d"),
            home_team=ht,
            away_team=at,
            ht_home_goals=ht_hg,
            ht_away_goals=ht_ag,
            sh_home_goals=sh_hg,
            sh_away_goals=sh_ag,
            ht_result=_result(ht_hg, ht_ag),
            ft_result=_result(ft_hg, ft_ag),
            ht_model_home=ht_grid.home_win,
            ht_model_draw=ht_grid.draw,
            ht_model_away=ht_grid.away_win,
            sh_model_home=sh_grid.home_win,
            sh_model_draw=sh_grid.draw,
            sh_model_away=sh_grid.away_win,
            ht_lambda_home=ht_grid.lambda_home,
            ht_lambda_away=ht_grid.lambda_away,
            sh_lambda_home=sh_grid.lambda_home,
            sh_lambda_away=sh_grid.lambda_away,
            htft_probs=htft_probs,
        )
        predictions.append(pred)

    return predictions, ht_packed, teams, sh_packed, teams
