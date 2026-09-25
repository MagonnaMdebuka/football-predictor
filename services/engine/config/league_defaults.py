"""Per-league hyperparameter configuration.

SPEC.md: "Every magic number lives in a per-league config table, not inline."

xi values are selected by maximising out-of-sample log-likelihood on a
validation slice (2022-23 and 2023-24 for the Premier League) and confirmed
once on the held-out test set. See ADR-015 and scripts/tune_xi.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueConfig:
    """Hyperparameters for a single league.

    Attributes:
        league_code: football-data.co.uk league code (e.g. "E0")
        league_name: human-readable league name
        xi: time-decay rate for Dixon-Coles weights
        rho_bounds: bounds for the Dixon-Coles dependence parameter
        max_goals: grid size for score-matrix predictions
        training_start_season: earliest season for training data
        validation_seasons: seasons used for hyperparameter selection
        test_seasons: held-out seasons for final evaluation
    """

    league_code: str
    league_name: str
    xi: float
    rho_bounds: tuple[float, float] = (-0.5, 0.5)
    max_goals: int = 11
    training_start_season: str = "2019-20"
    validation_seasons: tuple[str, ...] = ("2022-23", "2023-24")
    test_seasons: tuple[str, ...] = ("2024-25", "2025-26")
    has_corners: bool = False
    has_cards: bool = False
    ship_corners: bool = False
    ship_cards: bool = False


# Per-league config table.
# xi tuned on validation_seasons via scripts/tune_xi.py, confirmed on test_seasons.
LEAGUE_CONFIGS: dict[str, LeagueConfig] = {
    "E0": LeagueConfig(
        league_code="E0",
        league_name="Premier League",
        xi=0.0065,
        has_corners=True,
        has_cards=True,
    ),
}


def get_league_config(league_code: str) -> LeagueConfig:
    """Return the configuration for a league.

    Raises KeyError if the league has no configuration entry.
    """
    return LEAGUE_CONFIGS[league_code]
