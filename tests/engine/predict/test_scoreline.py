"""Tests for scoreline disagreement detection."""

import numpy as np
import pytest

from services.engine.models.grid import ScoreGrid
from services.engine.predict.scoreline import check_scoreline_disagreement


def _make_grid(home_goals: int, away_goals: int, modal_prob: float = 0.15) -> ScoreGrid:
    """Create a grid where (home_goals, away_goals) is the modal scoreline."""
    grid = np.full((11, 11), 0.005, dtype=np.float64)
    grid[home_goals, away_goals] = modal_prob
    grid /= grid.sum()  # normalise
    return ScoreGrid(
        grid=grid, home_team="Home", away_team="Away",
        lambda_home=1.5, lambda_away=1.2,
    )


def test_agreement_returns_none():
    """When modal scoreline agrees with 1X2 favourite → None."""
    # Modal = 1-0 (home win), and with that much mass on home-win cells,
    # the 1X2 favourite should be home.
    grid = np.zeros((11, 11), dtype=np.float64)
    # Make home wins dominant
    grid[1, 0] = 0.15
    grid[2, 0] = 0.10
    grid[2, 1] = 0.10
    grid[3, 1] = 0.05
    # Some draws and away wins
    grid[1, 1] = 0.08
    grid[0, 0] = 0.05
    grid[0, 1] = 0.04
    grid[1, 2] = 0.03
    # Fill remainder
    remaining = 1.0 - grid.sum()
    grid[0, 2] = remaining
    sg = ScoreGrid(grid=grid, home_team="Home", away_team="Away",
                   lambda_home=1.8, lambda_away=0.9)
    result = check_scoreline_disagreement(sg)
    assert result is None


def test_disagreement_returns_dict():
    """When modal scoreline implies draw but 1X2 favours home → disagreement."""
    # Start with uniform tiny background, then place specific masses
    grid = np.full((11, 11), 0.001, dtype=np.float64)
    # Modal is 1-1 (draw) at 10%
    grid[1, 1] = 0.10
    # Many home-win cells collectively outweigh the single draw
    grid[2, 1] = 0.09
    grid[1, 0] = 0.09
    grid[3, 1] = 0.08
    grid[2, 0] = 0.07
    grid[3, 0] = 0.06
    grid[4, 1] = 0.05
    grid[4, 2] = 0.04
    # Draws kept low
    grid[0, 0] = 0.03
    grid[2, 2] = 0.03
    # Away wins tiny
    grid[0, 1] = 0.01
    grid[1, 2] = 0.01
    # Normalise to sum to 1
    grid /= grid.sum()

    sg = ScoreGrid(grid=grid, home_team="Home", away_team="Away",
                   lambda_home=1.6, lambda_away=1.1)

    # Verify our setup: modal should be 1-1 (draw), but 1X2 favours home
    modal_h, modal_a, _ = sg.most_likely_score()
    assert (modal_h, modal_a) == (1, 1), f"Expected modal 1-1, got {modal_h}-{modal_a}"
    assert sg.home_win > sg.draw, "Expected home_win > draw for disagreement"

    result = check_scoreline_disagreement(sg)
    assert result is not None
    assert result["modal_result"] == "draw"
    assert result["favourite_result"] == "home"
    assert "modal_score" in result
    assert "message" in result


def test_draw_modal_and_draw_favourite_is_agreement():
    """When both modal scoreline and 1X2 indicate draw → None."""
    grid = np.zeros((11, 11), dtype=np.float64)
    # Draws dominate
    grid[1, 1] = 0.14
    grid[0, 0] = 0.10
    grid[2, 2] = 0.08
    # Home and away balanced
    grid[1, 0] = 0.06
    grid[2, 1] = 0.05
    grid[0, 1] = 0.06
    grid[1, 2] = 0.05
    remaining = 1.0 - grid.sum()
    grid[3, 3] = remaining

    sg = ScoreGrid(grid=grid, home_team="Home", away_team="Away",
                   lambda_home=1.3, lambda_away=1.3)
    result = check_scoreline_disagreement(sg)
    assert result is None
