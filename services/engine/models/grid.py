"""Score probability grid for match outcome prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from services.engine.models.params import DixonColesParams
from services.engine.models.poisson import goal_expectancy, poisson_pmf
from services.engine.models.tau import tau

MAX_GOALS = 11


@dataclass(frozen=True)
class ScoreGrid:
    """NxN matrix of score probabilities for a single match.

    grid[i, j] = P(home scores i, away scores j).

    Attributes:
        grid: (N, N) probability matrix where N = max_goals
        home_team: name of the home team
        away_team: name of the away team
        lambda_home: expected home goals
        lambda_away: expected away goals
    """

    grid: NDArray[np.float64]
    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float

    @property
    def home_win(self) -> float:
        """P(home win) — sum of lower triangle."""
        return float(np.tril(self.grid, k=-1).sum())

    @property
    def draw(self) -> float:
        """P(draw) — sum of diagonal."""
        return float(np.trace(self.grid))

    @property
    def away_win(self) -> float:
        """P(away win) — sum of upper triangle."""
        return float(np.triu(self.grid, k=1).sum())

    def predict_scoreline(self, home_goals: int, away_goals: int) -> float:
        """Return P(home=h, away=a) for a specific scoreline."""
        n = self.grid.shape[0]
        if home_goals >= n or away_goals >= n:
            return 0.0
        return float(self.grid[home_goals, away_goals])

    def most_likely_score(self) -> tuple[int, int, float]:
        """Return (home_goals, away_goals, probability) for the most likely scoreline."""
        idx = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        return int(idx[0]), int(idx[1]), float(self.grid[idx])


def build_grid(
    params: DixonColesParams,
    home_team: str,
    away_team: str,
    max_goals: int = MAX_GOALS,
) -> ScoreGrid:
    """Build an NxN score probability grid for a match.

    Combines independent Poisson probabilities with the Dixon-Coles
    tau correction for low-scoring outcomes.

    Args:
        params: fitted Dixon-Coles parameters
        home_team: home team name
        away_team: away team name
        max_goals: grid dimension (default 11 for FT, 7 for HT)
    """
    lam_h, lam_a = goal_expectancy(
        attack_h=params.team_attack(home_team),
        defence_a=params.team_defence(away_team),
        attack_a=params.team_attack(away_team),
        defence_h=params.team_defence(home_team),
        mu=params.mu,
        gamma=params.gamma,
    )
    lam_h_scalar = float(lam_h[0])
    lam_a_scalar = float(lam_a[0])

    grid = np.zeros((max_goals, max_goals), dtype=np.float64)

    for i in range(max_goals):
        for j in range(max_goals):
            p_home = float(poisson_pmf(i, lam_h_scalar))
            p_away = float(poisson_pmf(j, lam_a_scalar))
            tau_val = float(tau(i, j, lam_h_scalar, lam_a_scalar, params.rho)[0])
            grid[i, j] = p_home * p_away * tau_val

    return ScoreGrid(
        grid=grid,
        home_team=home_team,
        away_team=away_team,
        lambda_home=lam_h_scalar,
        lambda_away=lam_a_scalar,
    )
