"""Extract bookmaker odds from raw JSON and convert to probabilities.

Odds come from football-data.co.uk CSV rows stored as JSON in match_source_rows.raw.
Primary source: Pinnacle closing (PSCH/PSCD/PSCA).
Fallback: market average closing (AvgCH/AvgCD/AvgCA).
Overround is stripped via proportional normalisation.
"""

from __future__ import annotations

import json

import numpy as np
from numpy.typing import NDArray

from services.engine.backtest.types import BookmakerOddsCols


def extract_odds(
    raw_json: str,
    cols: BookmakerOddsCols | None = None,
) -> tuple[float, float, float] | None:
    """Extract decimal odds (home, draw, away) from a raw JSON string.

    Tries primary columns first (Pinnacle), falls back to market average.
    Returns None if neither source has valid odds.

    Args:
        raw_json: JSON string from match_source_rows.raw
        cols: column name configuration

    Returns:
        (odds_home, odds_draw, odds_away) or None if unavailable.
    """
    if cols is None:
        cols = BookmakerOddsCols()

    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return None

    # Try primary (Pinnacle)
    result = _try_extract(data, cols.primary_home, cols.primary_draw, cols.primary_away)
    if result is not None:
        return result

    # Try fallback (market average)
    return _try_extract(data, cols.fallback_home, cols.fallback_draw, cols.fallback_away)


def _try_extract(
    data: dict,
    col_home: str,
    col_draw: str,
    col_away: str,
) -> tuple[float, float, float] | None:
    """Try to extract valid odds from specific columns."""
    try:
        h = float(data[col_home])
        d = float(data[col_draw])
        a = float(data[col_away])
    except (KeyError, TypeError, ValueError):
        return None

    # Odds must be > 1.0 to be valid
    if h <= 1.0 or d <= 1.0 or a <= 1.0:
        return None

    return (h, d, a)


def odds_to_probabilities(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> tuple[float, float, float]:
    """Convert decimal odds to probabilities, stripping overround.

    Uses proportional normalisation: implied_prob = (1/odds) / sum(1/odds).

    Args:
        odds_home: decimal odds for home win
        odds_draw: decimal odds for draw
        odds_away: decimal odds for away win

    Returns:
        (p_home, p_draw, p_away) summing to 1.0.
    """
    implied_h = 1.0 / odds_home
    implied_d = 1.0 / odds_draw
    implied_a = 1.0 / odds_away
    total = implied_h + implied_d + implied_a

    return (implied_h / total, implied_d / total, implied_a / total)


def overround(odds_home: float, odds_draw: float, odds_away: float) -> float:
    """Calculate the bookmaker overround (vig/juice).

    Returns the total implied probability minus 1.0.
    A fair book has overround = 0.0.
    """
    return (1.0 / odds_home + 1.0 / odds_draw + 1.0 / odds_away) - 1.0


def extract_probabilities(
    raw_json: str,
    cols: BookmakerOddsCols | None = None,
) -> tuple[float, float, float] | None:
    """Extract odds and convert to normalised probabilities in one step.

    Returns None if odds are unavailable.
    """
    odds = extract_odds(raw_json, cols)
    if odds is None:
        return None
    return odds_to_probabilities(*odds)


def batch_extract_probabilities(
    raw_jsons: list[str | None],
    cols: BookmakerOddsCols | None = None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None, int]:
    """Extract probabilities from a batch of raw JSON strings.

    Returns:
        (p_home, p_draw, p_away, exclusion_count) where arrays are None if
        all matches lack odds. exclusion_count is the number of matches
        without valid odds.
    """
    if cols is None:
        cols = BookmakerOddsCols()

    homes = []
    draws = []
    aways = []
    exclusion_count = 0

    for raw in raw_jsons:
        if raw is None:
            exclusion_count += 1
            homes.append(np.nan)
            draws.append(np.nan)
            aways.append(np.nan)
            continue
        probs = extract_probabilities(raw, cols)
        if probs is None:
            exclusion_count += 1
            homes.append(np.nan)
            draws.append(np.nan)
            aways.append(np.nan)
        else:
            homes.append(probs[0])
            draws.append(probs[1])
            aways.append(probs[2])

    p_home = np.array(homes, dtype=np.float64)
    p_draw = np.array(draws, dtype=np.float64)
    p_away = np.array(aways, dtype=np.float64)

    if exclusion_count == len(raw_jsons):
        return None, None, None, exclusion_count

    return p_home, p_draw, p_away, exclusion_count
