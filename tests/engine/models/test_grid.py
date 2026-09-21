"""Tests for score probability grid."""

import numpy as np
import pytest

from services.engine.models.grid import MAX_GOALS, build_grid
from services.engine.models.params import DixonColesParams
from services.engine.models.poisson import goal_expectancy, poisson_pmf

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Spurs"]


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
def rho_zero_params() -> DixonColesParams:
    return DixonColesParams(
        teams=TEAMS,
        mu=0.25,
        attack=np.array([0.3, -0.1, 0.2, -0.4]),
        defence=np.array([-0.2, 0.1, 0.0, 0.1]),
        gamma=0.2,
        rho=0.0,
    )


class TestGridProperties:
    def test_shape(self, sample_params: DixonColesParams):
        sg = build_grid(sample_params, "Arsenal", "Chelsea")
        assert sg.grid.shape == (MAX_GOALS, MAX_GOALS)

    def test_sums_to_approximately_one(self, sample_params: DixonColesParams):
        sg = build_grid(sample_params, "Arsenal", "Chelsea")
        assert sg.grid.sum() == pytest.approx(1.0, abs=0.005)

    def test_non_negative(self, sample_params: DixonColesParams):
        sg = build_grid(sample_params, "Arsenal", "Chelsea")
        assert np.all(sg.grid >= 0)

    def test_1x2_sums_to_one(self, sample_params: DixonColesParams):
        sg = build_grid(sample_params, "Arsenal", "Chelsea")
        total = sg.home_win + sg.draw + sg.away_win
        assert total == pytest.approx(1.0, abs=0.005)


class TestRhoZeroMatchesPoisson:
    def test_grid_matches_independent_poisson(self, rho_zero_params: DixonColesParams):
        """With rho=0, grid should equal the product of independent Poisson PMFs."""
        sg = build_grid(rho_zero_params, "Arsenal", "Chelsea")
        lam_h, lam_a = goal_expectancy(
            attack_h=rho_zero_params.team_attack("Arsenal"),
            defence_a=rho_zero_params.team_defence("Chelsea"),
            attack_a=rho_zero_params.team_attack("Chelsea"),
            defence_h=rho_zero_params.team_defence("Arsenal"),
            mu=rho_zero_params.mu,
            gamma=rho_zero_params.gamma,
        )
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                expected = float(poisson_pmf(i, lam_h[0])) * float(poisson_pmf(j, lam_a[0]))
                assert sg.grid[i, j] == pytest.approx(expected, abs=1e-12)


class TestHomeFavourite:
    def test_strong_home_team_favoured(self, sample_params: DixonColesParams):
        """Arsenal (strong attack) at home vs Spurs (weak attack) should favour home."""
        sg = build_grid(sample_params, "Arsenal", "Spurs")
        assert sg.home_win > sg.away_win


class TestPredictScoreline:
    def test_specific_score(self, sample_params: DixonColesParams):
        sg = build_grid(sample_params, "Arsenal", "Chelsea")
        p = sg.predict_scoreline(1, 0)
        assert p == pytest.approx(sg.grid[1, 0])

    def test_out_of_range_returns_zero(self, sample_params: DixonColesParams):
        sg = build_grid(sample_params, "Arsenal", "Chelsea")
        assert sg.predict_scoreline(15, 0) == 0.0

    def test_consistency_with_grid(self, sample_params: DixonColesParams):
        """Sum of predict_scoreline over all cells should match grid.sum()."""
        sg = build_grid(sample_params, "Arsenal", "Chelsea")
        total = sum(
            sg.predict_scoreline(i, j)
            for i in range(MAX_GOALS)
            for j in range(MAX_GOALS)
        )
        assert total == pytest.approx(sg.grid.sum())


class TestMostLikelyScore:
    def test_returns_valid_score(self, sample_params: DixonColesParams):
        sg = build_grid(sample_params, "Arsenal", "Chelsea")
        h, a, p = sg.most_likely_score()
        assert 0 <= h < MAX_GOALS
        assert 0 <= a < MAX_GOALS
        assert p == pytest.approx(sg.grid[h, a])
