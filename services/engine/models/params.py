"""Dixon-Coles model parameters: dataclass, pack/unpack with sum-to-zero."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DixonColesParams:
    """Fitted Dixon-Coles model parameters.

    Attributes:
        teams: ordered list of team names (length n)
        mu: home-advantage intercept
        attack: attack strength per team (length n, sums to zero)
        defence: defence weakness per team (length n, sums to zero)
        gamma: overall scoring-rate parameter
        rho: dependence parameter for low-score correction
    """

    teams: list[str]
    mu: float
    attack: NDArray[np.float64]
    defence: NDArray[np.float64]
    gamma: float
    rho: float

    def team_attack(self, team: str) -> float:
        """Return attack strength for a team."""
        idx = self.teams.index(team)
        return float(self.attack[idx])

    def team_defence(self, team: str) -> float:
        """Return defence weakness for a team."""
        idx = self.teams.index(team)
        return float(self.defence[idx])


def pack(params: DixonColesParams) -> NDArray[np.float64]:
    """Flatten parameters into a 1-D vector for the optimiser.

    Vector layout (length 2n + 1):
        [mu, attack_0..attack_{n-2}, defence_0..defence_{n-2}, gamma, rho]

    The last team's attack/defence is omitted — derived as -sum(others)
    to enforce the sum-to-zero constraint without explicit scipy constraints.
    """
    n = len(params.teams)
    vec = np.empty(2 * n + 1, dtype=np.float64)
    vec[0] = params.mu
    vec[1 : n] = params.attack[: n - 1]
    vec[n : 2 * n - 1] = params.defence[: n - 1]
    vec[2 * n - 1] = params.gamma
    vec[2 * n] = params.rho
    return vec


def unpack(vec: NDArray[np.float64], teams: list[str]) -> DixonColesParams:
    """Reconstruct DixonColesParams from a flat vector and team list.

    Derives the last team's attack/defence as -sum(others).
    """
    n = len(teams)
    mu = float(vec[0])
    attack_free = vec[1:n]
    defence_free = vec[n : 2 * n - 1]
    gamma = float(vec[2 * n - 1])
    rho = float(vec[2 * n])

    attack = np.empty(n, dtype=np.float64)
    attack[: n - 1] = attack_free
    attack[n - 1] = -attack_free.sum()

    defence = np.empty(n, dtype=np.float64)
    defence[: n - 1] = defence_free
    defence[n - 1] = -defence_free.sum()

    return DixonColesParams(
        teams=teams,
        mu=mu,
        attack=attack,
        defence=defence,
        gamma=gamma,
        rho=rho,
    )


def vector_length(n_teams: int) -> int:
    """Return the expected length of the packed parameter vector."""
    return 2 * n_teams + 1
