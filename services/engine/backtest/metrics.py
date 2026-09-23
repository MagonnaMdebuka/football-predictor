"""Vectorised evaluation metrics for match outcome predictions.

All functions accept NumPy arrays of probabilities and actuals,
returning scalar metric values. Probabilities are clipped to [eps, 1-eps]
before log operations to avoid numerical issues.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-15


def ranked_probability_score(
    p_home: NDArray[np.float64],
    p_draw: NDArray[np.float64],
    p_away: NDArray[np.float64],
    actual: NDArray[np.str_],
) -> float:
    """Compute mean Ranked Probability Score for 1X2 outcomes.

    RPS penalises predictions that place probability mass far from the actual
    outcome, measured via cumulative distribution differences.

    Lower is better. Perfect = 0, worst = 1.

    Args:
        p_home: predicted P(home win) per match
        p_draw: predicted P(draw) per match
        p_away: predicted P(away win) per match
        actual: actual result per match ('H', 'D', or 'A')

    Returns:
        Mean RPS across all matches.
    """
    n = len(actual)
    if n == 0:
        return 0.0

    # Actual indicators
    o_home = (actual == "H").astype(np.float64)
    o_draw = (actual == "D").astype(np.float64)

    # Cumulative predicted
    cum_p1 = p_home
    cum_p2 = p_home + p_draw

    # Cumulative actual
    cum_o1 = o_home
    cum_o2 = o_home + o_draw

    # RPS = (1/2) * sum of squared cumulative differences
    rps_per_match = 0.5 * ((cum_p1 - cum_o1) ** 2 + (cum_p2 - cum_o2) ** 2)
    return float(np.mean(rps_per_match))


def log_loss(
    p_home: NDArray[np.float64],
    p_draw: NDArray[np.float64],
    p_away: NDArray[np.float64],
    actual: NDArray[np.str_],
) -> float:
    """Compute mean categorical log loss for 1X2 outcomes.

    Lower is better. Perfect = 0.

    Args:
        p_home: predicted P(home win) per match
        p_draw: predicted P(draw) per match
        p_away: predicted P(away win) per match
        actual: actual result per match ('H', 'D', or 'A')

    Returns:
        Mean log loss across all matches.
    """
    n = len(actual)
    if n == 0:
        return 0.0

    o_home = (actual == "H").astype(np.float64)
    o_draw = (actual == "D").astype(np.float64)
    o_away = (actual == "A").astype(np.float64)

    p_h = np.clip(p_home, _EPS, 1 - _EPS)
    p_d = np.clip(p_draw, _EPS, 1 - _EPS)
    p_a = np.clip(p_away, _EPS, 1 - _EPS)

    ll = -(o_home * np.log(p_h) + o_draw * np.log(p_d) + o_away * np.log(p_a))
    return float(np.mean(ll))


def brier_score(
    p: NDArray[np.float64],
    actual_indicator: NDArray[np.float64],
) -> float:
    """Compute mean Brier score for a single binary outcome.

    Lower is better. Perfect = 0, worst = 1.

    Args:
        p: predicted probability of the event
        actual_indicator: 1.0 if event occurred, 0.0 otherwise

    Returns:
        Mean Brier score.
    """
    n = len(actual_indicator)
    if n == 0:
        return 0.0
    return float(np.mean((p - actual_indicator) ** 2))


def hit_rate(
    p_home: NDArray[np.float64],
    p_draw: NDArray[np.float64],
    p_away: NDArray[np.float64],
    actual: NDArray[np.str_],
) -> float:
    """Compute hit rate: fraction of matches where the most likely outcome was correct.

    Args:
        p_home: predicted P(home win) per match
        p_draw: predicted P(draw) per match
        p_away: predicted P(away win) per match
        actual: actual result per match ('H', 'D', or 'A')

    Returns:
        Hit rate in [0, 1].
    """
    n = len(actual)
    if n == 0:
        return 0.0

    probs = np.stack([p_home, p_draw, p_away], axis=1)
    labels = np.array(["H", "D", "A"])
    predicted = labels[np.argmax(probs, axis=1)]
    return float(np.mean(predicted == actual))


def compute_metric_set(
    p_home: NDArray[np.float64],
    p_draw: NDArray[np.float64],
    p_away: NDArray[np.float64],
    actual: NDArray[np.str_],
) -> tuple[float, float, float, float, float, int, float]:
    """Compute all metrics at once, returning a tuple matching MetricSet fields.

    Returns:
        (rps, log_loss, brier_home, brier_draw, brier_away, n_matches, hit_rate)
    """
    o_home = (actual == "H").astype(np.float64)
    o_draw = (actual == "D").astype(np.float64)
    o_away = (actual == "A").astype(np.float64)

    return (
        ranked_probability_score(p_home, p_draw, p_away, actual),
        log_loss(p_home, p_draw, p_away, actual),
        brier_score(p_home, o_home),
        brier_score(p_draw, o_draw),
        brier_score(p_away, o_away),
        len(actual),
        hit_rate(p_home, p_draw, p_away, actual),
    )
