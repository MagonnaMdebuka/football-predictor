"""Count-derived markets from NB2 marginals (corners and booking points).

Over/under lines are computed by convolution of two NB marginals (home+away).
Team-level O/U use the individual marginals directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.stats import poisson as poisson_dist

from services.engine.models.negbin import negbin_pmf

# Default maximum k for PMF computation (sufficient for corners and booking points)
DEFAULT_MAX_K = 80

# Corner O/U lines
CORNER_MATCH_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
CORNER_TEAM_LINES = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

# Booking point O/U lines
BOOKING_MATCH_LINES = [20.5, 30.5, 40.5, 50.5, 60.5]
BOOKING_TEAM_LINES = [10.5, 15.5, 20.5, 25.5, 30.5]

# Gate-only lines: subset of match lines with informative base-rate splits.
# Extreme lines near 0/1 carry no information and inflate mean Brier noise.
CORNER_GATE_LINES = [8.5, 9.5, 10.5, 11.5, 12.5]
BOOKING_GATE_LINES = [20.5, 30.5, 40.5, 50.5]


def _r4(v: float) -> float:
    """Round to 4 decimal places for JSON output."""
    return round(float(v), 4)


@dataclass(frozen=True)
class CountMarkets:
    """Markets derived from a count model's NB2 marginals.

    Attributes:
        match_over_under: list of {line, over, under} for total (home+away)
        home_over_under: list of {line, over, under} for home team only
        away_over_under: list of {line, over, under} for away team only
    """

    match_over_under: list[dict]
    home_over_under: list[dict]
    away_over_under: list[dict]


def _compute_marginal_pmf(
    mu: float, alpha: float, max_k: int,
) -> np.ndarray:
    """Compute PMF vector P(X=k) for k=0..max_k."""
    ks = np.arange(max_k + 1)
    return negbin_pmf(ks, mu, alpha)


def _convolve_pmfs(
    pmf_home: np.ndarray, pmf_away: np.ndarray,
) -> np.ndarray:
    """Convolve two PMF vectors to get P(X+Y=k).

    Returns a vector of length len(pmf_home) + len(pmf_away) - 1.
    """
    return np.convolve(pmf_home, pmf_away)


def count_to_markets(
    mu_home: float,
    mu_away: float,
    alpha: float,
    match_lines: list[float],
    team_lines: list[float],
    max_k: int = DEFAULT_MAX_K,
) -> CountMarkets:
    """Derive over/under markets from NB2 marginals.

    Args:
        mu_home: expected count for home team
        mu_away: expected count for away team
        alpha: NB2 overdispersion parameter
        match_lines: O/U lines for total (home+away)
        team_lines: O/U lines for individual teams
        max_k: maximum k for PMF computation

    Returns:
        CountMarkets with match and team-level O/U probabilities.
    """
    # Compute marginal PMFs
    pmf_home = _compute_marginal_pmf(mu_home, alpha, max_k)
    pmf_away = _compute_marginal_pmf(mu_away, alpha, max_k)

    # Convolution for total
    pmf_total = _convolve_pmfs(pmf_home, pmf_away)

    # Build CDF for total
    cdf_total = np.cumsum(pmf_total)

    # Match over/under
    match_ou = []
    for line in match_lines:
        k = int(line)  # e.g. 9.5 -> 9, P(under) = P(X <= 9) = CDF(9)
        if k < len(cdf_total):
            p_under = float(cdf_total[k])
        else:
            p_under = 1.0
        p_over = 1.0 - p_under
        match_ou.append({
            "line": line,
            "over": _r4(p_over),
            "under": _r4(p_under),
        })

    # Team over/under (home)
    cdf_home = np.cumsum(pmf_home)
    home_ou = []
    for line in team_lines:
        k = int(line)
        if k < len(cdf_home):
            p_under = float(cdf_home[k])
        else:
            p_under = 1.0
        home_ou.append({
            "line": line,
            "over": _r4(1.0 - p_under),
            "under": _r4(p_under),
        })

    # Team over/under (away)
    cdf_away = np.cumsum(pmf_away)
    away_ou = []
    for line in team_lines:
        k = int(line)
        if k < len(cdf_away):
            p_under = float(cdf_away[k])
        else:
            p_under = 1.0
        away_ou.append({
            "line": line,
            "over": _r4(1.0 - p_under),
            "under": _r4(p_under),
        })

    return CountMarkets(
        match_over_under=match_ou,
        home_over_under=home_ou,
        away_over_under=away_ou,
    )


def total_count_to_match_markets(
    mu_total: float,
    alpha_total: float,
    match_lines: list[float],
    max_k: int = DEFAULT_MAX_K,
) -> list[dict]:
    """Derive match-level O/U markets from a single NB2 distribution on totals.

    Unlike count_to_markets() which convolves two marginals, this uses a
    directly-fitted total distribution — avoiding the independence assumption.

    Args:
        mu_total: expected total count (fitted directly)
        alpha_total: NB2 overdispersion for totals
        match_lines: O/U lines for the total
        max_k: maximum k for PMF computation

    Returns:
        List of {line, over, under} dicts.
    """
    pmf = _compute_marginal_pmf(mu_total, alpha_total, max_k)
    cdf = np.cumsum(pmf)

    match_ou = []
    for line in match_lines:
        k = int(line)
        if k < len(cdf):
            p_under = float(cdf[k])
        else:
            p_under = 1.0
        p_over = 1.0 - p_under
        match_ou.append({
            "line": line,
            "over": _r4(p_over),
            "under": _r4(p_under),
        })

    return match_ou


def _booking_point_pmf(
    mu_yellow: float,
    alpha_yellow: float,
    mu_red: float,
    max_yellow: int = 15,
    max_red: int = 5,
) -> np.ndarray:
    """Compute a single team's booking-point PMF from yellow/red distributions.

    Booking points = 10*Y + 25*R.
    P(Y=y) from NB2(mu_yellow, alpha_yellow), P(R=r) from Poisson(mu_red).
    Joint (y,r) -> bp = 10*y + 25*r, accumulate into PMF array.

    Returns an array of length 10*max_yellow + 25*max_red + 1 (= 276 by default).
    """
    max_bp = 10 * max_yellow + 25 * max_red
    bp_pmf = np.zeros(max_bp + 1)

    # Yellow PMF: NB2
    ks_y = np.arange(max_yellow + 1)
    pmf_y = negbin_pmf(ks_y, mu_yellow, alpha_yellow)

    # Red PMF: Poisson
    ks_r = np.arange(max_red + 1)
    pmf_r = np.array([float(poisson_dist.pmf(r, mu_red)) for r in ks_r])

    for y in range(max_yellow + 1):
        for r in range(max_red + 1):
            bp = 10 * y + 25 * r
            if bp <= max_bp:
                bp_pmf[bp] += float(pmf_y[y]) * float(pmf_r[r])

    return bp_pmf


def compound_booking_to_markets(
    mu_yellow_home: float,
    mu_yellow_away: float,
    alpha_yellow: float,
    mu_red_home: float,
    mu_red_away: float,
    match_lines: list[float],
    team_lines: list[float],
    max_yellow: int = 15,
    max_red: int = 5,
) -> CountMarkets:
    """Derive booking-point O/U markets from compound yellow/red distributions.

    Each team's booking-point PMF is built from NB2(yellow) * Poisson(red),
    mapping joint (y,r) -> 10*y + 25*r. Home and away PMFs are then convolved
    for match-total O/U lines.

    Args:
        mu_yellow_home: expected yellow cards for home team
        mu_yellow_away: expected yellow cards for away team
        alpha_yellow: NB2 overdispersion for yellows
        mu_red_home: expected red cards for home team
        mu_red_away: expected red cards for away team
        match_lines: O/U lines for total booking points
        team_lines: O/U lines for individual team booking points
        max_yellow: maximum yellow cards per team
        max_red: maximum red cards per team

    Returns:
        CountMarkets with match and team-level O/U probabilities.
    """
    # Build per-team booking-point PMFs
    bp_pmf_home = _booking_point_pmf(
        mu_yellow_home, alpha_yellow, mu_red_home, max_yellow, max_red,
    )
    bp_pmf_away = _booking_point_pmf(
        mu_yellow_away, alpha_yellow, mu_red_away, max_yellow, max_red,
    )

    # Convolve for match total
    bp_pmf_total = np.convolve(bp_pmf_home, bp_pmf_away)

    # CDFs
    cdf_total = np.cumsum(bp_pmf_total)
    cdf_home = np.cumsum(bp_pmf_home)
    cdf_away = np.cumsum(bp_pmf_away)

    # Match O/U
    match_ou = []
    for line in match_lines:
        k = int(line)
        if k < len(cdf_total):
            p_under = float(cdf_total[k])
        else:
            p_under = 1.0
        match_ou.append({
            "line": line,
            "over": _r4(1.0 - p_under),
            "under": _r4(p_under),
        })

    # Home team O/U
    home_ou = []
    for line in team_lines:
        k = int(line)
        if k < len(cdf_home):
            p_under = float(cdf_home[k])
        else:
            p_under = 1.0
        home_ou.append({
            "line": line,
            "over": _r4(1.0 - p_under),
            "under": _r4(p_under),
        })

    # Away team O/U
    away_ou = []
    for line in team_lines:
        k = int(line)
        if k < len(cdf_away):
            p_under = float(cdf_away[k])
        else:
            p_under = 1.0
        away_ou.append({
            "line": line,
            "over": _r4(1.0 - p_under),
            "under": _r4(p_under),
        })

    return CountMarkets(
        match_over_under=match_ou,
        home_over_under=home_ou,
        away_over_under=away_ou,
    )
