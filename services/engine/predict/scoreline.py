"""Scoreline disagreement detection.

Identifies cases where the most likely individual scoreline implies a
different result from the overall 1X2 favourite. This is expected (not
an error) and provides useful context for users.
"""

from __future__ import annotations

import numpy as np

from services.engine.models.grid import ScoreGrid


def _result_from_score(home: int, away: int) -> str:
    if home > away:
        return "home"
    elif home < away:
        return "away"
    return "draw"


def _favourite_result(home_prob: float, draw_prob: float, away_prob: float) -> str:
    best = max(home_prob, draw_prob, away_prob)
    if best == home_prob:
        return "home"
    elif best == away_prob:
        return "away"
    return "draw"


def check_scoreline_disagreement(grid: ScoreGrid) -> dict | None:
    """Check whether the modal scoreline disagrees with the 1X2 favourite.

    Returns None if they agree, or a dict with disagreement details.
    """
    home_goals, away_goals, modal_prob = grid.most_likely_score()
    modal_result = _result_from_score(home_goals, away_goals)
    fav_result = _favourite_result(grid.home_win, grid.draw, grid.away_win)

    if modal_result == fav_result:
        return None

    score_str = f"{home_goals}-{away_goals}"
    fav_pct = max(grid.home_win, grid.draw, grid.away_win)

    return {
        "modal_score": score_str,
        "modal_result": modal_result,
        "favourite_result": fav_result,
        "message": (
            f"Most likely score {score_str} ({modal_prob:.0%}) implies "
            f"{modal_result}, but overall probability favours "
            f"{fav_result} win ({fav_pct:.0%}). Many {fav_result}-win "
            f"scores collectively outweigh the single most likely "
            f"{modal_result}."
        ),
    }
