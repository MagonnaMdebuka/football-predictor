"""Count model backtest harness — fit and predict for one refit date.

Fits an NB2 count model (corners or cards) on training data and generates
CountPrediction records for held-out matches at a single refit date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from services.engine.backtest.count_types import CountPrediction
from services.engine.markets.counts import (
    BOOKING_MATCH_LINES,
    BOOKING_TEAM_LINES,
    CORNER_MATCH_LINES,
    CORNER_TEAM_LINES,
    compound_booking_to_markets,
    count_to_markets,
    total_count_to_match_markets,
)
from services.engine.models.count_fit import fit_count_model
from services.engine.models.count_params import CountModelParams, pack
from services.engine.models.negbin import count_expectancy
from services.engine.models.total_count_fit import fit_total_count_model
from services.engine.models.total_count_params import (
    TotalCountModelParams,
    pack_total,
)


def _augment_count_params(
    params: CountModelParams,
    teams_needed: list[str],
) -> CountModelParams:
    """Add missing teams at league-average strength (attack=0, defence=0)."""
    missing = [t for t in teams_needed if t not in params.teams]
    if not missing:
        return params

    new_teams = list(params.teams) + missing
    new_attack = np.concatenate([params.attack, np.zeros(len(missing))])
    new_defence = np.concatenate([params.defence, np.zeros(len(missing))])

    return CountModelParams(
        teams=new_teams,
        mu=params.mu,
        attack=new_attack,
        defence=new_defence,
        gamma=params.gamma,
        alpha=params.alpha,
        referees=params.referees,
        referee_effect=params.referee_effect,
    )


def _augment_total_count_params(
    params: TotalCountModelParams,
    teams_needed: list[str],
) -> TotalCountModelParams:
    """Add missing teams at league-average effect (home_effect=0, away_effect=0)."""
    missing = [t for t in teams_needed if t not in params.teams]
    if not missing:
        return params

    new_teams = list(params.teams) + missing
    new_home = np.concatenate([params.home_effect, np.zeros(len(missing))])
    new_away = np.concatenate([params.away_effect, np.zeros(len(missing))])

    return TotalCountModelParams(
        teams=new_teams,
        mu=params.mu,
        home_effect=new_home,
        away_effect=new_away,
        alpha=params.alpha,
    )


def run_count_predictions(
    training_df: pd.DataFrame,
    prediction_matches: pd.DataFrame,
    model_type: str,
    weights: NDArray[np.float64] | None = None,
    include_referees: bool = False,
    min_referee_matches: int = 20,
    prev_packed: NDArray[np.float64] | None = None,
    prev_teams: list[str] | None = None,
    prev_total_packed: NDArray[np.float64] | None = None,
    prev_total_teams: list[str] | None = None,
) -> tuple[
    list[CountPrediction],
    NDArray[np.float64], list[str],
    NDArray[np.float64] | None, list[str] | None,
]:
    """Fit a count model and predict for one refit date's matches.

    For corners, fits both per-team and total models (dual architecture):
    - Per-team model: used for team-level O/U markets
    - Total model: used for match-level O/U markets (avoids independence assumption)

    For cards, fits per-team model only.

    Args:
        training_df: training data (strictly before prediction date)
        prediction_matches: held-out matches to predict
        model_type: 'corners' or 'cards'
        weights: optional time-decay weights for training data
        include_referees: whether to include referee effects (cards only)
        min_referee_matches: minimum matches for referee effect
        prev_packed: previous fit's packed vector for warm-start (per-team model)
        prev_teams: previous fit's team list for warm-start (per-team model)
        prev_total_packed: previous fit's packed vector for warm-start (total model)
        prev_total_teams: previous fit's team list for warm-start (total model)

    Returns:
        (predictions, packed_params, teams, total_packed, total_teams)
        for warm-start chaining. total_packed/total_teams are None for cards.
    """
    empty: tuple[
        list[CountPrediction],
        NDArray[np.float64], list[str],
        NDArray[np.float64] | None, list[str] | None,
    ] = ([], np.array([]), [], None, None)

    # Determine target columns and market lines
    if model_type == "corners":
        home_col, away_col = "home_corners", "away_corners"
        match_lines = CORNER_MATCH_LINES
        team_lines = CORNER_TEAM_LINES
    else:
        home_col, away_col = "home_booking_points", "away_booking_points"
        match_lines = BOOKING_MATCH_LINES
        team_lines = BOOKING_TEAM_LINES

    # Check that target columns exist
    if home_col not in training_df.columns or away_col not in training_df.columns:
        return empty

    # Drop rows with missing count data
    training_clean = training_df.dropna(subset=[home_col, away_col])
    if len(training_clean) < 10:
        return empty

    # Warm-start: use previous params if team set unchanged
    teams = sorted(
        set(training_clean["home_team"].unique()) | set(training_clean["away_team"].unique())
    )
    x0 = None
    if prev_packed is not None and prev_teams is not None and teams == prev_teams:
        x0 = prev_packed

    # Fit per-team model
    fit_result = fit_count_model(
        training_clean,
        target_home_col=home_col,
        target_away_col=away_col,
        weights=weights,
        include_referees=include_referees,
        min_referee_matches=min_referee_matches,
        x0=x0,
    )

    packed = pack(fit_result.params)

    # Fit total model (corners only)
    total_fit_result = None
    total_packed_out: NDArray[np.float64] | None = None
    total_teams_out: list[str] | None = None

    if model_type == "corners":
        t_x0 = None
        if (prev_total_packed is not None and prev_total_teams is not None
                and teams == prev_total_teams):
            t_x0 = prev_total_packed

        total_fit_result = fit_total_count_model(
            training_clean,
            target_home_col=home_col,
            target_away_col=away_col,
            weights=weights,
            x0=t_x0,
        )
        total_packed_out = pack_total(total_fit_result.params)
        total_teams_out = teams

    # Generate predictions
    predictions: list[CountPrediction] = []

    for _, match in prediction_matches.iterrows():
        ht = match["home_team"]
        at = match["away_team"]

        params = _augment_count_params(fit_result.params, [ht, at])

        hi = params.teams.index(ht)
        ai = params.teams.index(at)

        # Determine referee effect
        ref_name = None
        ref_eff = 0.0
        if include_referees and "referee" in match.index and match["referee"] is not None:
            ref_name = str(match["referee"])
            if params.referees and ref_name in params.referees:
                ref_idx = params.referees.index(ref_name)
                ref_eff = params.referee_effect[ref_idx]

        # Compute expected counts (per-team model)
        mu_h, mu_a = count_expectancy(
            attack_h=params.attack[hi],
            defence_a=params.defence[ai],
            attack_a=params.attack[ai],
            defence_h=params.defence[hi],
            mu=params.mu,
            gamma=params.gamma,
            ref_effect_h=ref_eff,
            ref_effect_a=ref_eff,
        )

        mu_h_val = mu_h.item()
        mu_a_val = mu_a.item()

        # Compute team-level markets from per-team model
        team_markets = count_to_markets(
            mu_h_val, mu_a_val, float(params.alpha),
            match_lines, team_lines,
        )

        # Match-level O/U: use total model for corners, per-team convolution otherwise
        mu_total_val: float | None = None
        alpha_total_val: float | None = None

        if total_fit_result is not None:
            total_params = _augment_total_count_params(
                total_fit_result.params, [ht, at],
            )
            t_hi = total_params.teams.index(ht)
            t_ai = total_params.teams.index(at)
            mu_total_val = float(np.exp(
                total_params.mu
                + total_params.home_effect[t_hi]
                + total_params.away_effect[t_ai]
            ))
            alpha_total_val = float(total_params.alpha)
            match_ou = total_count_to_match_markets(
                mu_total_val, alpha_total_val, match_lines,
            )
        else:
            match_ou = team_markets.match_over_under

        # Get actual counts (may be NaN for future matches)
        actual_h = int(match[home_col]) if pd.notna(match.get(home_col)) else 0
        actual_a = int(match[away_col]) if pd.notna(match.get(away_col)) else 0

        pred = CountPrediction(
            date=pd.Timestamp(match["date"]).strftime("%Y-%m-%d"),
            season=str(match["season"]),
            home_team=ht,
            away_team=at,
            model_type=model_type,
            actual_home=actual_h,
            actual_away=actual_a,
            mu_home=round(mu_h_val, 6),
            mu_away=round(mu_a_val, 6),
            alpha=round(float(params.alpha), 6),
            referee=ref_name,
            match_over_under=match_ou,
            home_over_under=team_markets.home_over_under,
            away_over_under=team_markets.away_over_under,
            n_training_matches=fit_result.n_matches,
            mu_total=round(mu_total_val, 6) if mu_total_val is not None else None,
            alpha_total=round(alpha_total_val, 6) if alpha_total_val is not None else None,
        )
        predictions.append(pred)

    return predictions, packed, teams, total_packed_out, total_teams_out


def run_compound_card_predictions(
    training_df: pd.DataFrame,
    prediction_matches: pd.DataFrame,
    weights: NDArray[np.float64] | None = None,
    min_referee_matches: int = 20,
    prev_yellow_packed: NDArray[np.float64] | None = None,
    prev_yellow_teams: list[str] | None = None,
    prev_red_packed: NDArray[np.float64] | None = None,
    prev_red_teams: list[str] | None = None,
) -> tuple[
    list[CountPrediction],
    NDArray[np.float64], list[str],
    NDArray[np.float64], list[str],
]:
    """Fit compound card model (yellows NB2 + reds Poisson) and predict.

    Returns:
        (predictions, yellow_packed, yellow_teams, red_packed, red_teams)
        for warm-start chaining.
    """
    empty = ([], np.array([]), [], np.array([]), [])

    # Check required columns
    for col in ("home_yellows", "away_yellows", "home_reds", "away_reds",
                "home_booking_points", "away_booking_points"):
        if col not in training_df.columns:
            return empty

    training_clean = training_df.dropna(
        subset=["home_yellows", "away_yellows", "home_reds", "away_reds",
                "home_booking_points", "away_booking_points"],
    )
    if len(training_clean) < 10:
        return empty

    # Team list for warm-start comparison
    teams = sorted(
        set(training_clean["home_team"].unique())
        | set(training_clean["away_team"].unique())
    )

    # Fit yellow NB2 model
    y_x0 = None
    if (prev_yellow_packed is not None and prev_yellow_teams is not None
            and teams == prev_yellow_teams):
        y_x0 = prev_yellow_packed

    yellow_fit = fit_count_model(
        training_clean,
        target_home_col="home_yellows",
        target_away_col="away_yellows",
        weights=weights,
        include_referees=True,
        min_referee_matches=min_referee_matches,
        x0=y_x0,
    )
    yellow_packed = pack(yellow_fit.params)

    # Fit red Poisson model (alpha forced near zero)
    r_x0 = None
    if (prev_red_packed is not None and prev_red_teams is not None
            and teams == prev_red_teams):
        r_x0 = prev_red_packed

    red_fit = fit_count_model(
        training_clean,
        target_home_col="home_reds",
        target_away_col="away_reds",
        weights=weights,
        include_referees=True,
        min_referee_matches=min_referee_matches,
        alpha_bounds=(1e-12, 1e-12),
        x0=r_x0,
    )
    red_packed = pack(red_fit.params)

    # Generate predictions
    predictions: list[CountPrediction] = []
    match_lines = BOOKING_MATCH_LINES
    team_lines = BOOKING_TEAM_LINES

    for _, match in prediction_matches.iterrows():
        ht = match["home_team"]
        at = match["away_team"]

        # Augment params for both models
        y_params = _augment_count_params(yellow_fit.params, [ht, at])
        r_params = _augment_count_params(red_fit.params, [ht, at])

        hi_y = y_params.teams.index(ht)
        ai_y = y_params.teams.index(at)
        hi_r = r_params.teams.index(ht)
        ai_r = r_params.teams.index(at)

        # Referee effect
        ref_name = None
        y_ref_eff = 0.0
        r_ref_eff = 0.0
        if "referee" in match.index and match["referee"] is not None:
            ref_name = str(match["referee"])
            if y_params.referees and ref_name in y_params.referees:
                y_ref_eff = y_params.referee_effect[y_params.referees.index(ref_name)]
            if r_params.referees and ref_name in r_params.referees:
                r_ref_eff = r_params.referee_effect[r_params.referees.index(ref_name)]

        # Yellow expectancies
        mu_y_h, mu_y_a = count_expectancy(
            attack_h=y_params.attack[hi_y],
            defence_a=y_params.defence[ai_y],
            attack_a=y_params.attack[ai_y],
            defence_h=y_params.defence[hi_y],
            mu=y_params.mu,
            gamma=y_params.gamma,
            ref_effect_h=y_ref_eff,
            ref_effect_a=y_ref_eff,
        )
        mu_y_h_val = mu_y_h.item()
        mu_y_a_val = mu_y_a.item()

        # Red expectancies
        mu_r_h, mu_r_a = count_expectancy(
            attack_h=r_params.attack[hi_r],
            defence_a=r_params.defence[ai_r],
            attack_a=r_params.attack[ai_r],
            defence_h=r_params.defence[hi_r],
            mu=r_params.mu,
            gamma=r_params.gamma,
            ref_effect_h=r_ref_eff,
            ref_effect_a=r_ref_eff,
        )
        mu_r_h_val = mu_r_h.item()
        mu_r_a_val = mu_r_a.item()

        # Compound markets
        markets = compound_booking_to_markets(
            mu_yellow_home=mu_y_h_val,
            mu_yellow_away=mu_y_a_val,
            alpha_yellow=float(y_params.alpha),
            mu_red_home=mu_r_h_val,
            mu_red_away=mu_r_a_val,
            match_lines=match_lines,
            team_lines=team_lines,
        )

        # Expected booking points per team: 10*E[Y] + 25*E[R]
        mu_bp_home = 10.0 * mu_y_h_val + 25.0 * mu_r_h_val
        mu_bp_away = 10.0 * mu_y_a_val + 25.0 * mu_r_a_val

        # Actual booking points
        actual_h = int(match["home_booking_points"]) if pd.notna(
            match.get("home_booking_points")) else 0
        actual_a = int(match["away_booking_points"]) if pd.notna(
            match.get("away_booking_points")) else 0

        pred = CountPrediction(
            date=pd.Timestamp(match["date"]).strftime("%Y-%m-%d"),
            season=str(match["season"]),
            home_team=ht,
            away_team=at,
            model_type="cards",
            actual_home=actual_h,
            actual_away=actual_a,
            mu_home=round(mu_bp_home, 6),
            mu_away=round(mu_bp_away, 6),
            alpha=round(float(y_params.alpha), 6),
            referee=ref_name,
            match_over_under=markets.match_over_under,
            home_over_under=markets.home_over_under,
            away_over_under=markets.away_over_under,
            n_training_matches=yellow_fit.n_matches,
            mu_yellow_home=round(mu_y_h_val, 6),
            mu_yellow_away=round(mu_y_a_val, 6),
            alpha_yellow=round(float(y_params.alpha), 6),
            mu_red_home=round(mu_r_h_val, 6),
            mu_red_away=round(mu_r_a_val, 6),
        )
        predictions.append(pred)

    return predictions, yellow_packed, teams, red_packed, teams
