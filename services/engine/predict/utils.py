"""Pure utility functions for the predict pipeline.

No database or heavy framework imports — safe for unit testing.
"""

from __future__ import annotations

import hashlib
import zlib
from datetime import date

import numpy as np


# ── Grid compression ──────────────────────────────────────────────────


def compress_grid(grid: np.ndarray) -> bytes:
    """Compress an 11x11 score grid for DB storage."""
    return zlib.compress(grid.astype(np.float64).tobytes())


def decompress_grid(data: bytes) -> np.ndarray:
    """Decompress a stored grid back to an 11x11 float64 array."""
    return np.frombuffer(zlib.decompress(data), dtype=np.float64).reshape(11, 11)


# ── Fingerprint ───────────────────────────────────────────────────────


def compute_fingerprint(
    league_id: int,
    n_matches: int,
    latest_date: date,
    model_version: str = "",
    git_commit: str = "",
) -> str:
    """SHA-256 fingerprint for idempotency checks.

    Includes model version and git commit so that code changes
    (even without new data) produce fresh predictions.
    """
    raw = (
        f"{league_id}:{n_matches}:{latest_date.isoformat()}"
        f":{model_version}:{git_commit}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Market flattening ─────────────────────────────────────────────────


def flatten_markets(markets_dict: dict, prediction_id: int) -> list[dict]:
    """Convert nested grid_to_markets output to flat MarketPrediction rows."""
    rows: list[dict] = []

    def _add(market: str, selection: str, probability: float, line: float | None = None):
        rows.append({
            "prediction_id": prediction_id,
            "market": market,
            "selection": selection,
            "probability": round(probability, 6),
            "line": line,
        })

    # match_result
    for sel, prob in markets_dict["match_result"].items():
        _add("match_result", sel, prob)

    # double_chance
    for sel, prob in markets_dict["double_chance"].items():
        _add("double_chance", sel, prob)

    # draw_no_bet
    for sel, prob in markets_dict["draw_no_bet"].items():
        _add("draw_no_bet", sel, prob)

    # over_under (list of dicts with line)
    for entry in markets_dict["over_under"]:
        _add("over_under", "over", entry["over"], line=entry["line"])
        _add("over_under", "under", entry["under"], line=entry["line"])

    # btts
    for sel, prob in markets_dict["btts"].items():
        _add("btts", sel, prob)

    # btts_over_25
    for sel, prob in markets_dict["btts_over_25"].items():
        _add("btts_over_25", sel, prob)

    # exact_total_goals
    for sel, prob in markets_dict["exact_total_goals"].items():
        _add("exact_total_goals", sel, prob)

    # correct_score
    for sel, prob in markets_dict["correct_score"].items():
        _add("correct_score", sel, prob)

    # winning_margin
    for sel, prob in markets_dict["winning_margin"].items():
        _add("winning_margin", sel, prob)

    # odd_even
    for sel, prob in markets_dict["odd_even"].items():
        _add("odd_even", sel, prob)

    # team_totals
    for entry in markets_dict["team_totals"]["home"]:
        _add("team_totals_home", "over", entry["over"], line=entry["line"])
        _add("team_totals_home", "under", entry["under"], line=entry["line"])
    for entry in markets_dict["team_totals"]["away"]:
        _add("team_totals_away", "over", entry["over"], line=entry["line"])
        _add("team_totals_away", "under", entry["under"], line=entry["line"])

    # clean_sheet
    for sel, prob in markets_dict["clean_sheet"].items():
        _add("clean_sheet", sel, prob)

    # win_to_nil
    for sel, prob in markets_dict["win_to_nil"].items():
        _add("win_to_nil", sel, prob)

    # european_handicap (list of dicts with line)
    for entry in markets_dict["european_handicap"]:
        _add("european_handicap", "home", entry["home"], line=entry["line"])
        _add("european_handicap", "draw", entry["draw"], line=entry["line"])
        _add("european_handicap", "away", entry["away"], line=entry["line"])

    # asian_handicap (list of dicts with line)
    for entry in markets_dict["asian_handicap"]:
        _add("asian_handicap", "home", entry["home"], line=entry["line"])
        _add("asian_handicap", "push", entry["push"], line=entry["line"])
        _add("asian_handicap", "away", entry["away"], line=entry["line"])

    return rows
