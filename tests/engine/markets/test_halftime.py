"""Tests for half-time markets, HT/FT market, and first goal timing."""

from __future__ import annotations

import numpy as np
import pytest

from services.engine.markets.halftime import (
    first_goal_timing,
    ht_grid_to_markets,
    htft_market,
)
from services.engine.models.grid import ScoreGrid, build_grid
from services.engine.models.params import DixonColesParams

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Spurs"]


@pytest.fixture
def ht_params() -> DixonColesParams:
    """Lower lambdas typical of half-time scoring rates (lambda ~ 0.5-0.8)."""
    return DixonColesParams(
        teams=TEAMS,
        mu=-0.60,
        attack=np.array([0.15, -0.05, 0.10, -0.20]),
        defence=np.array([-0.10, 0.05, 0.0, 0.05]),
        gamma=0.10,
        rho=-0.05,
    )


@pytest.fixture
def ht_grid(ht_params: DixonColesParams) -> ScoreGrid:
    return build_grid(ht_params, "Arsenal", "Chelsea", max_goals=7)


@pytest.fixture
def sh_grid(ht_params: DixonColesParams) -> ScoreGrid:
    return build_grid(ht_params, "Arsenal", "Chelsea", max_goals=7)


class TestHtGridToMarkets:
    def test_returns_ht_prefixed_market_types(self, ht_grid: ScoreGrid):
        markets = ht_grid_to_markets(ht_grid)
        types = {m["market"] for m in markets}
        assert "ht_match_result" in types
        assert "ht_over_under_0.5" in types
        assert "ht_btts" in types
        assert "ht_correct_score" in types

    def test_ht_1x2_sums_to_one(self, ht_grid: ScoreGrid):
        markets = ht_grid_to_markets(ht_grid)
        mr = [m for m in markets if m["market"] == "ht_match_result"]
        total = sum(m["probability"] for m in mr)
        assert total == pytest.approx(1.0, abs=5e-3)

    def test_ht_ou_lines_are_restricted(self, ht_grid: ScoreGrid):
        markets = ht_grid_to_markets(ht_grid)
        ou_markets = [m for m in markets if m["market"].startswith("ht_over_under")]
        lines = {m["market"] for m in ou_markets}
        assert "ht_over_under_0.5" in lines
        assert "ht_over_under_1.5" in lines
        assert "ht_over_under_2.5" in lines
        # 3.5+ should not be present for HT
        assert "ht_over_under_3.5" not in lines
        assert "ht_over_under_4.5" not in lines

    def test_ht_ou_over_under_sum_to_one(self, ht_grid: ScoreGrid):
        markets = ht_grid_to_markets(ht_grid)
        ou_05 = [m for m in markets if m["market"] == "ht_over_under_0.5"]
        assert len(ou_05) == 2  # over and under
        total = sum(m["probability"] for m in ou_05)
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_ht_btts_sums_to_one(self, ht_grid: ScoreGrid):
        markets = ht_grid_to_markets(ht_grid)
        btts = [m for m in markets if m["market"] == "ht_btts"]
        total = sum(m["probability"] for m in btts)
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_ht_correct_score_covers_up_to_3_3(self, ht_grid: ScoreGrid):
        markets = ht_grid_to_markets(ht_grid)
        cs = [m for m in markets if m["market"] == "ht_correct_score"]
        selections = {m["selection"] for m in cs}
        assert "0-0" in selections
        assert "1-1" in selections
        assert "3-3" in selections

    def test_ht_team_totals_lines_restricted(self, ht_grid: ScoreGrid):
        markets = ht_grid_to_markets(ht_grid)
        tt = [m for m in markets if m["market"].startswith("ht_home_total")]
        lines = {m["market"] for m in tt}
        assert "ht_home_total_0.5" in lines
        assert "ht_home_total_1.5" in lines
        # 2.5+ should not be present for HT team totals
        assert "ht_home_total_2.5" not in lines


class TestHtftMarket:
    def test_returns_9_outcomes(self, ht_grid: ScoreGrid, sh_grid: ScoreGrid):
        markets = htft_market(ht_grid, sh_grid)
        assert len(markets) == 9

    def test_sums_to_one(self, ht_grid: ScoreGrid, sh_grid: ScoreGrid):
        markets = htft_market(ht_grid, sh_grid)
        total = sum(m["probability"] for m in markets)
        assert total == pytest.approx(1.0, abs=5e-3)

    def test_ht_marginals_consistent(self, ht_grid: ScoreGrid, sh_grid: ScoreGrid):
        """Sum of HH+HD+HA should equal P(HT=H)."""
        markets = htft_market(ht_grid, sh_grid)
        probs = {m["selection"]: m["probability"] for m in markets}
        p_ht_home = probs["HH"] + probs["HD"] + probs["HA"]
        assert p_ht_home == pytest.approx(ht_grid.home_win, abs=1e-3)


class TestFirstGoalTiming:
    def test_higher_lambda_means_earlier_goal(self):
        low = first_goal_timing(0.5, 0.5)
        high = first_goal_timing(1.5, 1.5)
        p_low = next(m for m in low if m["selection"] == "before_15")
        p_high = next(m for m in high if m["selection"] == "before_15")
        assert p_high["probability"] > p_low["probability"]

    def test_probabilities_increase_with_time(self):
        markets = first_goal_timing(1.0, 1.0)
        p15 = next(m for m in markets if m["selection"] == "before_15")
        p30 = next(m for m in markets if m["selection"] == "before_30")
        p45 = next(m for m in markets if m["selection"] == "before_45")
        assert p15["probability"] < p30["probability"] < p45["probability"]

    def test_very_low_lambda(self):
        markets = first_goal_timing(0.05, 0.05)
        p15 = next(m for m in markets if m["selection"] == "before_15")
        assert p15["probability"] < 0.05
