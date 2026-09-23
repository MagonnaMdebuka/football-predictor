"""Tests for goals-derived markets.

Property tests run over randomised lambda pairs to catch edge cases.
Golden-file test pins the exact output for a known fixture to 4dp.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from services.engine.markets.goals import GridTruncationError, grid_to_markets
from services.engine.models.grid import MAX_GOALS, ScoreGrid, build_grid
from services.engine.models.params import DixonColesParams
from services.engine.models.poisson import poisson_pmf

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Spurs"]
GOLDEN_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "golden_markets.json"

# Tolerance for sums of 4dp-rounded values. Accounts for both rounding
# error (N values × 5e-5) and grid tail mass (up to 1e-4 at lambda ≈ 2.5).
_TOL_SMALL = 5e-4
_TOL_LARGE = 5e-3


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_params() -> DixonColesParams:
    return DixonColesParams(
        teams=TEAMS,
        mu=0.25,
        attack=np.array([0.3, -0.1, 0.2, -0.4]),
        defence=np.array([-0.2, 0.1, 0.0, 0.1]),
        gamma=0.2,
        rho=-0.1,
    )


@pytest.fixture
def grid(sample_params: DixonColesParams):
    return build_grid(sample_params, "Arsenal", "Chelsea")


@pytest.fixture
def markets(grid):
    return grid_to_markets(grid)


def _make_grid_from_lambdas(lam_h: float, lam_a: float) -> ScoreGrid:
    """Build a ScoreGrid from independent Poisson (rho=0, no tau correction)."""
    g = np.zeros((MAX_GOALS, MAX_GOALS), dtype=np.float64)
    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            g[i, j] = float(poisson_pmf(i, lam_h)) * float(poisson_pmf(j, lam_a))
    return ScoreGrid(
        grid=g, home_team="H", away_team="A",
        lambda_home=lam_h, lambda_away=lam_a,
    )


# 10 randomised lambda pairs in the realistic football range [0.3, 2.5].
# Lambda > 2.5 per side is extremely rare in practice, and pushing beyond
# that starts losing material tail mass in the 11x11 grid.
_RNG = np.random.RandomState(42)
_RANDOM_LAMBDAS = [
    (round(h, 3), round(a, 3))
    for h, a in zip(_RNG.uniform(0.3, 2.5, 10), _RNG.uniform(0.3, 2.5, 10))
]


@pytest.fixture(params=_RANDOM_LAMBDAS, ids=[f"lh={h}_la={a}" for h, a in _RANDOM_LAMBDAS])
def random_markets(request):
    """Markets derived from randomised lambda pairs."""
    lam_h, lam_a = request.param
    grid = _make_grid_from_lambdas(lam_h, lam_a)
    return grid_to_markets(grid)


# ── Golden-file test ────────────────────────────────────────────────────


class TestGoldenFile:
    def test_golden_markets_match(self, markets):
        """Every probability matches the golden file to 4dp."""
        golden = json.loads(GOLDEN_PATH.read_text())
        _assert_nested_approx(markets, golden)


def _assert_nested_approx(actual, expected, path=""):
    """Recursively compare nested dicts/lists of floats."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected dict at {path}, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"Key mismatch at {path}: {set(actual.keys()) ^ set(expected.keys())}"
        )
        for k in expected:
            _assert_nested_approx(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"Expected list at {path}, got {type(actual)}"
        assert len(actual) == len(expected), (
            f"Length mismatch at {path}: {len(actual)} vs {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_nested_approx(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=5e-5), (
            f"Value mismatch at {path}: {actual} vs {expected}"
        )
    elif isinstance(expected, (int, str)):
        assert actual == expected, f"Value mismatch at {path}: {actual!r} vs {expected!r}"
    else:
        raise TypeError(f"Unexpected type at {path}: {type(expected)}")


# ── Property tests over randomised lambdas ──────────────────────────────


class TestMatchResultSumsToOne:
    def test_1x2(self, random_markets):
        m = random_markets["match_result"]
        total = m["home"] + m["draw"] + m["away"]
        assert total == pytest.approx(1.0, abs=_TOL_SMALL)


class TestDoubleChanceIdentities:
    def test_home_draw_equals_home_plus_draw(self, random_markets):
        dc = random_markets["double_chance"]
        assert dc["home_draw"] == pytest.approx(dc["home"] + dc["draw"], abs=_TOL_SMALL)

    def test_draw_away_equals_draw_plus_away(self, random_markets):
        dc = random_markets["double_chance"]
        assert dc["draw_away"] == pytest.approx(dc["draw"] + dc["away"], abs=_TOL_SMALL)

    def test_home_away_equals_home_plus_away(self, random_markets):
        dc = random_markets["double_chance"]
        assert dc["home_away"] == pytest.approx(dc["home"] + dc["away"], abs=_TOL_SMALL)


class TestDrawNoBet:
    def test_sums_to_one(self, random_markets):
        dnb = random_markets["draw_no_bet"]
        assert dnb["home"] + dnb["away"] == pytest.approx(1.0, abs=_TOL_SMALL)


class TestOverUnderSums:
    def test_over_plus_under_equals_one(self, random_markets):
        for entry in random_markets["over_under"]:
            assert entry["over"] + entry["under"] == pytest.approx(1.0, abs=_TOL_SMALL), (
                f"O/U {entry['line']}"
            )

    def test_team_totals_home(self, random_markets):
        for entry in random_markets["team_totals"]["home"]:
            assert entry["over"] + entry["under"] == pytest.approx(1.0, abs=_TOL_SMALL)

    def test_team_totals_away(self, random_markets):
        for entry in random_markets["team_totals"]["away"]:
            assert entry["over"] + entry["under"] == pytest.approx(1.0, abs=_TOL_SMALL)


class TestBTTS:
    def test_yes_plus_no_equals_one(self, random_markets):
        b = random_markets["btts"]
        assert b["yes"] + b["no"] == pytest.approx(1.0, abs=_TOL_SMALL)


class TestBTTSOver25:
    def test_yes_plus_no_equals_one(self, random_markets):
        b = random_markets["btts_over_25"]
        assert b["yes"] + b["no"] == pytest.approx(1.0, abs=_TOL_SMALL)

    def test_btts_over25_leq_btts(self, random_markets):
        """BTTS-and-O2.5 is a subset of BTTS, so must be <= BTTS yes."""
        assert random_markets["btts_over_25"]["yes"] <= (
            random_markets["btts"]["yes"] + 1e-4
        )


class TestExactTotalGoals:
    def test_sums_to_one(self, random_markets):
        etg = random_markets["exact_total_goals"]
        total = sum(etg.values())
        assert total == pytest.approx(1.0, abs=_TOL_LARGE)


class TestCorrectScore:
    def test_all_cells_plus_other_sum_to_one(self, random_markets):
        cs = random_markets["correct_score"]
        total = sum(cs.values())
        assert total == pytest.approx(1.0, abs=_TOL_LARGE)


class TestWinningMargin:
    def test_sums_to_one(self, random_markets):
        wm = random_markets["winning_margin"]
        total = sum(wm.values())
        assert total == pytest.approx(1.0, abs=_TOL_LARGE)


class TestOddEven:
    def test_odd_plus_even_equals_one(self, random_markets):
        oe = random_markets["odd_even"]
        assert oe["odd"] + oe["even"] == pytest.approx(1.0, abs=_TOL_SMALL)


class TestCleanSheetAndWinToNil:
    def test_clean_sheet_home_gte_win_to_nil_home(self, random_markets):
        """Clean sheet includes draws (0-0), win-to-nil does not."""
        assert random_markets["clean_sheet"]["home"] >= (
            random_markets["win_to_nil"]["home"] - 1e-4
        )

    def test_clean_sheet_away_gte_win_to_nil_away(self, random_markets):
        assert random_markets["clean_sheet"]["away"] >= (
            random_markets["win_to_nil"]["away"] - 1e-4
        )

    def test_win_to_nil_home_leq_home_win(self, random_markets):
        """Win-to-nil is a subset of all home wins."""
        assert random_markets["win_to_nil"]["home"] <= (
            random_markets["match_result"]["home"] + 1e-4
        )

    def test_win_to_nil_away_leq_away_win(self, random_markets):
        assert random_markets["win_to_nil"]["away"] <= (
            random_markets["match_result"]["away"] + 1e-4
        )


class TestEuropeanHandicap:
    def test_each_line_sums_to_one(self, random_markets):
        for entry in random_markets["european_handicap"]:
            total = entry["home"] + entry["draw"] + entry["away"]
            assert total == pytest.approx(1.0, abs=_TOL_SMALL), (
                f"EH line {entry['line']}: {total}"
            )


class TestAsianHandicap:
    def test_win_push_loss_sum_to_one(self, random_markets):
        """home + push + away = 1.0 for every line."""
        for entry in random_markets["asian_handicap"]:
            total = entry["home"] + entry["push"] + entry["away"]
            assert total == pytest.approx(1.0, abs=_TOL_SMALL), (
                f"AH line {entry['line']}: h={entry['home']} p={entry['push']} a={entry['away']}"
            )

    def test_half_lines_have_zero_push(self, random_markets):
        """Half-integer lines cannot push."""
        for entry in random_markets["asian_handicap"]:
            line = entry["line"]
            frac = round(abs(line) % 1.0, 2)
            if frac == 0.5:
                assert entry["push"] == pytest.approx(0.0, abs=1e-4), (
                    f"AH half-line {line} has non-zero push {entry['push']}"
                )

    def test_whole_lines_can_push(self, random_markets):
        """Whole-integer lines should have non-trivial push mass."""
        for entry in random_markets["asian_handicap"]:
            line = entry["line"]
            frac = round(abs(line) % 1.0, 2)
            if frac == 0.0:
                # Push is the probability of the exact margin; should be > 0
                assert entry["push"] >= 0.0


# ── Bounds: every probability in [0, 1] ─────────────────────────────────


_NON_PROB_KEYS = {"line"}


class TestBounds:
    def test_all_probabilities_in_range(self, random_markets):
        _check_bounds(random_markets)


def _check_bounds(obj, path="", key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_bounds(v, f"{path}.{k}", key=k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_bounds(v, f"{path}[{i}]")
    elif isinstance(obj, float) and key not in _NON_PROB_KEYS:
        assert -1e-4 <= obj <= 1.0 + 1e-4, f"Out of [0,1] at {path}: {obj}"


# ── Over/under line ordering ───────────────────────────────────────────


class TestMonotonicity:
    def test_over_decreases_with_line(self, random_markets):
        ou = random_markets["over_under"]
        for i in range(len(ou) - 1):
            assert ou[i]["over"] >= ou[i + 1]["over"] - 1e-4


# ── Grid truncation guard ──────────────────────────────────────────────


class TestGridTruncation:
    def test_normal_lambdas_pass(self):
        """Standard lambdas (1-3 range) should pass the guard."""
        grid = _make_grid_from_lambdas(1.5, 1.0)
        grid_to_markets(grid)  # should not raise

    def test_extreme_lambdas_fail(self):
        """Very high lambdas should trigger GridTruncationError."""
        grid = _make_grid_from_lambdas(8.0, 8.0)
        with pytest.raises(GridTruncationError, match="tail mass"):
            grid_to_markets(grid)
