"""Total count model parameters: dataclass, pack/unpack with sum-to-zero.

Fits total counts directly (e.g. total corners = home + away) as a single
NB2 distribution with per-team home/away effects:
    mu_total = exp(mu + home_effect[h] + away_effect[a])

Unlike the per-team model (count_params.py), this model has no gamma, attack,
or defence — only venue-specific team effects on the match total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TotalCountModelParams:
    """Fitted total count model parameters.

    Attributes:
        teams: ordered list of team names (length n)
        mu: league total-rate intercept
        home_effect: team i's contribution to total when playing at home (length n, sums to zero)
        away_effect: team j's contribution to total when playing away (length n, sums to zero)
        alpha: NB2 overdispersion (variance = mu + alpha * mu^2)
    """

    teams: list[str]
    mu: float
    home_effect: NDArray[np.float64]
    away_effect: NDArray[np.float64]
    alpha: float

    def team_home_effect(self, team: str) -> float:
        """Return home effect for a team."""
        idx = self.teams.index(team)
        return float(self.home_effect[idx])

    def team_away_effect(self, team: str) -> float:
        """Return away effect for a team."""
        idx = self.teams.index(team)
        return float(self.away_effect[idx])


def pack_total(params: TotalCountModelParams) -> NDArray[np.float64]:
    """Flatten parameters into a 1-D vector for the optimiser.

    Vector layout:
        [home_0..home_{n-2}, away_0..away_{n-2}, mu, alpha]

    Length: 2*n where n = len(teams).

    The last team's home_effect and away_effect are omitted — derived as
    -sum(others) to enforce sum-to-zero.
    """
    n = len(params.teams)
    vec_len = total_vector_length(n)
    vec = np.empty(vec_len, dtype=np.float64)

    vec[:n - 1] = params.home_effect[:n - 1]
    vec[n - 1:2 * (n - 1)] = params.away_effect[:n - 1]
    vec[2 * (n - 1)] = params.mu
    vec[2 * (n - 1) + 1] = params.alpha

    return vec


def unpack_total(
    vec: NDArray[np.float64],
    teams: list[str],
) -> TotalCountModelParams:
    """Reconstruct TotalCountModelParams from a flat vector.

    Derives the last team's home_effect and away_effect as -sum(others).
    """
    n = len(teams)

    home_free = vec[:n - 1]
    away_free = vec[n - 1:2 * (n - 1)]
    mu = float(vec[2 * (n - 1)])
    alpha = float(vec[2 * (n - 1) + 1])

    home_effect = np.empty(n, dtype=np.float64)
    home_effect[:n - 1] = home_free
    home_effect[n - 1] = -home_free.sum()

    away_effect = np.empty(n, dtype=np.float64)
    away_effect[:n - 1] = away_free
    away_effect[n - 1] = -away_free.sum()

    return TotalCountModelParams(
        teams=teams,
        mu=mu,
        home_effect=home_effect,
        away_effect=away_effect,
        alpha=alpha,
    )


def total_vector_length(n_teams: int) -> int:
    """Return the expected length of the packed parameter vector.

    Length = 2*(n-1) + 2 = 2*n
    """
    return 2 * n_teams
