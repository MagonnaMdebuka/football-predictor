"""Goals-derived betting markets from score grids."""

from services.engine.markets.goals import GridTruncationError, grid_to_markets

__all__ = ["GridTruncationError", "grid_to_markets"]
