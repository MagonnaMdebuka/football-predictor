"""Goals-derived markets from a ScoreGrid probability matrix.

All 15 market types are derived from the 11x11 score grid. Each market is
computed by a private helper; the public entry point is ``grid_to_markets``.
"""

from __future__ import annotations

import numpy as np

from services.engine.models.grid import ScoreGrid

# Maximum tail mass allowed beyond the grid boundary (guards high-lambda grids).
# With MAX_GOALS=11, Poisson(3.5) loses ~0.03% per side; 1e-3 accommodates
# lambdas up to ~4 per side while catching truly extreme expectancies.
_MAX_TAIL_MASS = 1e-3

# Lines for over/under and team totals
_OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
_TEAM_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]

# European handicap lines (integer shifts)
_EH_LINES = list(range(-3, 4))  # -3 to +3

# Asian handicap lines: -2.5 to +2.5 in 0.25 steps
_AH_LINES = [round(x * 0.25, 2) for x in range(-10, 11)]


def _r4(v: float) -> float:
    """Round to 4 decimal places for JSON output."""
    return round(float(v), 4)


class GridTruncationError(ValueError):
    """Raised when a grid's tail mass exceeds the safety threshold."""


def _check_tail_mass(grid: ScoreGrid) -> None:
    """Assert the raw grid captures enough probability mass.

    The 11x11 grid truncates at 10 goals per side. For normal lambdas this
    loses < 1e-6 of mass. If lambdas are too high the tail becomes material
    and all derived markets would be silently wrong.
    """
    grid_sum = float(grid.grid.sum())
    tail_mass = 1.0 - grid_sum
    if tail_mass > _MAX_TAIL_MASS:
        raise GridTruncationError(
            f"Grid tail mass {tail_mass:.2e} exceeds threshold {_MAX_TAIL_MASS:.0e}. "
            f"lambda_home={grid.lambda_home:.4f}, lambda_away={grid.lambda_away:.4f}. "
            f"The grid is too small for these goal expectancies."
        )


def grid_to_markets(grid: ScoreGrid) -> dict:
    """Derive all goals-based markets from a score probability grid.

    Returns a JSON-serialisable nested dict with keys for each market type.

    Raises GridTruncationError if the grid's truncated tail mass exceeds 1e-6.
    """
    _check_tail_mass(grid)

    g = grid.grid  # (11, 11) numpy array
    n = g.shape[0]

    return {
        "match_result": _match_result(g),
        "double_chance": _double_chance(g),
        "draw_no_bet": _draw_no_bet(g),
        "over_under": _over_under(g, n),
        "btts": _btts(g),
        "btts_over_25": _btts_over_25(g),
        "exact_total_goals": _exact_total_goals(g, n),
        "correct_score": _correct_score(g, n),
        "winning_margin": _winning_margin(g, n),
        "odd_even": _odd_even(g, n),
        "team_totals": _team_totals(g, n),
        "clean_sheet": _clean_sheet(g),
        "win_to_nil": _win_to_nil(g),
        "european_handicap": _european_handicap(g, n),
        "asian_handicap": _asian_handicap(g, n),
    }


# ── 1. Match result (1X2) ──────────────────────────────────────────────


def _match_result(g: np.ndarray) -> dict:
    home = float(np.tril(g, k=-1).sum())
    draw = float(np.trace(g))
    away = float(np.triu(g, k=1).sum())
    return {"home": _r4(home), "draw": _r4(draw), "away": _r4(away)}


# ── 2. Double chance ───────────────────────────────────────────────────


def _double_chance(g: np.ndarray) -> dict:
    home = float(np.tril(g, k=-1).sum())
    draw = float(np.trace(g))
    away = float(np.triu(g, k=1).sum())
    return {
        "home_draw": _r4(home + draw),
        "draw_away": _r4(draw + away),
        "home_away": _r4(home + away),
        "home": _r4(home),
        "draw": _r4(draw),
        "away": _r4(away),
    }


# ── 3. Draw no bet ─────────────────────────────────────────────────────


def _draw_no_bet(g: np.ndarray) -> dict:
    home = float(np.tril(g, k=-1).sum())
    away = float(np.triu(g, k=1).sum())
    total = home + away
    return {
        "home": _r4(home / total),
        "away": _r4(away / total),
    }


# ── 4. Over/under 0.5–5.5 ──────────────────────────────────────────────


def _over_under(g: np.ndarray, n: int) -> list[dict]:
    results = []
    for line in _OU_LINES:
        over = 0.0
        for i in range(n):
            for j in range(n):
                if i + j > line:
                    over += g[i, j]
        results.append({
            "line": line,
            "over": _r4(over),
            "under": _r4(1.0 - over),
        })
    return results


# ── 5. BTTS ─────────────────────────────────────────────────────────────


def _btts(g: np.ndarray) -> dict:
    yes = float(g[1:, 1:].sum())
    return {"yes": _r4(yes), "no": _r4(1.0 - yes)}


# ── 6. BTTS and over 2.5 ───────────────────────────────────────────────


def _btts_over_25(g: np.ndarray) -> dict:
    """P(both teams score AND total goals > 2.5)."""
    n = g.shape[0]
    yes = 0.0
    for i in range(1, n):
        for j in range(1, n):
            if i + j > 2:
                yes += g[i, j]
    return {"yes": _r4(float(yes)), "no": _r4(1.0 - float(yes))}


# ── 7. Exact total goals (0, 1, 2, 3, 4, 5, 6+) ───────────────────────


def _exact_total_goals(g: np.ndarray, n: int) -> dict:
    """P(total goals = k) for k in 0..5 plus a 6+ bucket."""
    totals = {}
    for k in range(6):
        prob = 0.0
        for i in range(min(k + 1, n)):
            j = k - i
            if 0 <= j < n:
                prob += g[i, j]
        totals[str(k)] = _r4(float(prob))

    # 6+ is the remainder
    exact_sum = sum(totals[str(k)] for k in range(6))
    totals["6+"] = _r4(1.0 - exact_sum)
    return totals


# ── 8. Correct score ───────────────────────────────────────────────────


def _correct_score(g: np.ndarray, n: int) -> dict:
    scores: list[tuple[str, float]] = []
    for i in range(n):
        for j in range(n):
            scores.append((f"{i}-{j}", float(g[i, j])))

    # Sort by probability descending, take top 12
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:12]
    other = sum(p for _, p in scores[12:])

    result = {}
    for label, prob in top:
        result[label] = _r4(prob)
    result["other"] = _r4(other)
    return result


# ── 9. Winning margin ──────────────────────────────────────────────────


def _winning_margin(g: np.ndarray, n: int) -> dict:
    margins: dict[str, float] = {}

    # Home wins: margin = i - j > 0
    for m in range(1, n):
        total = 0.0
        for i in range(n):
            j = i - m
            if 0 <= j < n:
                total += g[i, j]
        margins[f"home_+{m}"] = total

    # Draw
    margins["draw"] = float(np.trace(g))

    # Away wins: margin = j - i > 0
    for m in range(1, n):
        total = 0.0
        for j in range(n):
            i = j - m
            if 0 <= i < n:
                total += g[i, j]
        margins[f"away_+{m}"] = total

    return {k: _r4(v) for k, v in margins.items()}


# ── 10. Odd/even total goals ───────────────────────────────────────────


def _odd_even(g: np.ndarray, n: int) -> dict:
    odd = 0.0
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 1:
                odd += g[i, j]
    return {"odd": _r4(odd), "even": _r4(1.0 - odd)}


# ── 11. Team totals (home/away over/under 0.5–4.5) ─────────────────────


def _team_totals(g: np.ndarray, n: int) -> dict:
    home_marginal = g.sum(axis=1)  # sum over columns → P(home=i)
    away_marginal = g.sum(axis=0)  # sum over rows → P(away=j)

    home_results = []
    away_results = []

    for line in _TEAM_LINES:
        threshold = int(line + 0.5)  # 0.5→1, 1.5→2, etc.

        h_over = float(home_marginal[threshold:].sum())
        home_results.append({
            "line": line,
            "over": _r4(h_over),
            "under": _r4(1.0 - h_over),
        })

        a_over = float(away_marginal[threshold:].sum())
        away_results.append({
            "line": line,
            "over": _r4(a_over),
            "under": _r4(1.0 - a_over),
        })

    return {"home": home_results, "away": away_results}


# ── 12. Clean sheet ────────────────────────────────────────────────────


def _clean_sheet(g: np.ndarray) -> dict:
    home_cs = float(g[:, 0].sum())  # away scores 0
    away_cs = float(g[0, :].sum())  # home scores 0
    return {"home": _r4(home_cs), "away": _r4(away_cs)}


# ── 13. Win to nil ─────────────────────────────────────────────────────


def _win_to_nil(g: np.ndarray) -> dict:
    # Home win to nil: home scores > 0 AND away scores 0
    home_wtn = float(g[1:, 0].sum())
    # Away win to nil: home scores 0 AND away scores > 0
    away_wtn = float(g[0, 1:].sum())
    return {"home": _r4(home_wtn), "away": _r4(away_wtn)}


# ── 14. European handicap ──────────────────────────────────────────────


def _european_handicap(g: np.ndarray, n: int) -> list[dict]:
    results = []
    for hcap in _EH_LINES:
        home = 0.0
        draw = 0.0
        away = 0.0
        for i in range(n):
            for j in range(n):
                adjusted = (i + hcap) - j  # home goals + handicap - away goals
                if adjusted > 0:
                    home += g[i, j]
                elif adjusted == 0:
                    draw += g[i, j]
                else:
                    away += g[i, j]
        results.append({
            "line": hcap,
            "home": _r4(home),
            "draw": _r4(draw),
            "away": _r4(away),
        })
    return results


# ── 15. Asian handicap ─────────────────────────────────────────────────


def _asian_handicap(g: np.ndarray, n: int) -> list[dict]:
    results = []
    for line in _AH_LINES:
        win, push, loss = _ah_line(g, n, line)
        results.append({
            "line": line,
            "home": _r4(win),
            "push": _r4(push),
            "away": _r4(loss),
        })
    return results


def _ah_line(
    g: np.ndarray, n: int, line: float,
) -> tuple[float, float, float]:
    """Compute Asian handicap probabilities for a single line.

    Returns (home_win, push, away_win) as raw probabilities.

    - Half-integer lines (e.g. -0.5, +1.5): push is always 0.
    - Whole-integer lines (e.g. 0, +1): push is the diagonal mass.
    - Quarter lines (e.g. -0.25, +0.75): half the stake goes on each
      adjacent half-integer line. The push is half the whole-line push.
    """
    frac = round(abs(line) % 1.0, 2)
    is_quarter = frac in (0.25, 0.75)

    if is_quarter:
        lower = round(line - 0.25, 2)
        upper = round(line + 0.25, 2)
        h1, p1, a1 = _ah_raw(g, n, lower)
        h2, p2, a2 = _ah_raw(g, n, upper)
        return (h1 + h2) / 2, (p1 + p2) / 2, (a1 + a2) / 2
    else:
        return _ah_raw(g, n, line)


def _ah_raw(
    g: np.ndarray, n: int, line: float,
) -> tuple[float, float, float]:
    """Raw Asian handicap for a half-integer or whole-integer line.

    Returns (home_win, push, away_win). For half-integer lines push is 0.
    """
    home = 0.0
    away = 0.0
    push = 0.0

    for i in range(n):
        for j in range(n):
            margin = (i - j) + line
            if margin > 1e-9:
                home += g[i, j]
            elif margin < -1e-9:
                away += g[i, j]
            else:
                push += g[i, j]

    return home, push, away
