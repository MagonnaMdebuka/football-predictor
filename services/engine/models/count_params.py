"""Count model parameters: dataclass, pack/unpack with sum-to-zero.

Mirrors DixonColesParams but adds alpha (NB2 overdispersion) and optional
referee effects for the cards model. No rho/tau — corners and cards have
no low-score dependence structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CountModelParams:
    """Fitted count model parameters (corners or cards).

    Attributes:
        teams: ordered list of team names (length n)
        mu: league count-rate intercept
        attack: attack strength per team (length n, sums to zero)
        defence: defence weakness per team (length n, sums to zero)
        gamma: home-advantage parameter
        alpha: NB2 overdispersion (variance = mu + alpha * mu^2)
        referees: ordered list of referee names (length r); empty for corners
        referee_effect: per-referee offset (length r, sums to zero); empty for corners
    """

    teams: list[str]
    mu: float
    attack: NDArray[np.float64]
    defence: NDArray[np.float64]
    gamma: float
    alpha: float
    referees: list[str] = field(default_factory=list)
    referee_effect: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    def team_attack(self, team: str) -> float:
        """Return attack strength for a team."""
        idx = self.teams.index(team)
        return float(self.attack[idx])

    def team_defence(self, team: str) -> float:
        """Return defence weakness for a team."""
        idx = self.teams.index(team)
        return float(self.defence[idx])

    def ref_effect(self, referee: str) -> float:
        """Return referee effect offset. Returns 0.0 if referee not found."""
        if referee not in self.referees:
            return 0.0
        idx = self.referees.index(referee)
        return float(self.referee_effect[idx])


def pack(params: CountModelParams) -> NDArray[np.float64]:
    """Flatten parameters into a 1-D vector for the optimiser.

    Vector layout:
        [gamma, attack_0..attack_{n-2}, defence_0..defence_{n-2}, mu, alpha]
        optionally followed by [ref_0..ref_{r-2}]

    Length: 2n + 2 + max(r-1, 0) where n = len(teams), r = len(referees).

    The last team's attack/defence (and last referee's effect) are omitted —
    derived as -sum(others) to enforce sum-to-zero.
    """
    n = len(params.teams)
    r = len(params.referees)
    vec_len = vector_length(n, r)
    vec = np.empty(vec_len, dtype=np.float64)

    vec[0] = params.gamma
    vec[1:n] = params.attack[: n - 1]
    vec[n: 2 * n - 1] = params.defence[: n - 1]
    vec[2 * n - 1] = params.mu
    vec[2 * n] = params.alpha

    if r > 1:
        vec[2 * n + 1: 2 * n + 1 + r - 1] = params.referee_effect[: r - 1]

    return vec


def unpack(
    vec: NDArray[np.float64],
    teams: list[str],
    referees: list[str] | None = None,
) -> CountModelParams:
    """Reconstruct CountModelParams from a flat vector.

    Derives the last team's attack/defence (and last referee's effect) as
    -sum(others).
    """
    if referees is None:
        referees = []
    n = len(teams)
    r = len(referees)

    gamma = float(vec[0])
    attack_free = vec[1:n]
    defence_free = vec[n: 2 * n - 1]
    mu = float(vec[2 * n - 1])
    alpha = float(vec[2 * n])

    attack = np.empty(n, dtype=np.float64)
    attack[: n - 1] = attack_free
    attack[n - 1] = -attack_free.sum()

    defence = np.empty(n, dtype=np.float64)
    defence[: n - 1] = defence_free
    defence[n - 1] = -defence_free.sum()

    referee_effect = np.array([], dtype=np.float64)
    if r > 1:
        ref_free = vec[2 * n + 1: 2 * n + 1 + r - 1]
        referee_effect = np.empty(r, dtype=np.float64)
        referee_effect[: r - 1] = ref_free
        referee_effect[r - 1] = -ref_free.sum()
    elif r == 1:
        referee_effect = np.zeros(1, dtype=np.float64)

    return CountModelParams(
        teams=teams,
        mu=mu,
        attack=attack,
        defence=defence,
        gamma=gamma,
        alpha=alpha,
        referees=referees,
        referee_effect=referee_effect,
    )


def vector_length(n_teams: int, n_referees: int = 0) -> int:
    """Return the expected length of the packed parameter vector.

    Length = 2*n + 2 + max(r-1, 0)
    """
    return 2 * n_teams + 2 + max(n_referees - 1, 0)
