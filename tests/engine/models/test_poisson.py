"""Tests for Poisson goal model."""

import numpy as np
import pytest

from services.engine.models.poisson import goal_expectancy, poisson_log_pmf, poisson_pmf


class TestGoalExpectancy:
    def test_home_advantage_increases_lambda(self):
        """Positive gamma should give lambda_home > lambda_away for equal teams."""
        lam_h, lam_a = goal_expectancy(
            attack_h=0.0, defence_a=0.0,
            attack_a=0.0, defence_h=0.0,
            mu=0.0, gamma=0.3,
        )
        assert lam_h[0] > lam_a[0]

    def test_symmetric_when_no_home_advantage(self):
        """With gamma=0 and identical team params, lambda_home == lambda_away."""
        lam_h, lam_a = goal_expectancy(
            attack_h=0.1, defence_a=-0.1,
            attack_a=0.1, defence_h=-0.1,
            mu=0.2, gamma=0.0,
        )
        assert lam_h[0] == pytest.approx(lam_a[0])

    def test_stronger_attack_increases_goals(self):
        """Higher attack strength should produce higher lambda."""
        lam_h_weak, _ = goal_expectancy(
            attack_h=0.0, defence_a=0.0,
            attack_a=0.0, defence_h=0.0,
            mu=0.2, gamma=0.1,
        )
        lam_h_strong, _ = goal_expectancy(
            attack_h=0.5, defence_a=0.0,
            attack_a=0.0, defence_h=0.0,
            mu=0.2, gamma=0.1,
        )
        assert lam_h_strong[0] > lam_h_weak[0]

    def test_vectorised_input(self):
        """Should handle arrays of team parameters."""
        attack_h = np.array([0.1, 0.2, 0.3])
        defence_a = np.array([0.0, -0.1, 0.1])
        attack_a = np.array([0.0, 0.1, -0.1])
        defence_h = np.array([0.1, 0.0, 0.2])
        lam_h, lam_a = goal_expectancy(attack_h, defence_a, attack_a, defence_h, mu=0.2, gamma=0.1)
        assert lam_h.shape == (3,)
        assert lam_a.shape == (3,)

    def test_positive_lambdas(self):
        """Lambdas must always be positive (exp never returns zero)."""
        lam_h, lam_a = goal_expectancy(
            attack_h=-5.0, defence_a=-5.0,
            attack_a=-5.0, defence_h=-5.0,
            mu=-5.0, gamma=-5.0,
        )
        assert lam_h[0] > 0
        assert lam_a[0] > 0


class TestPoissonPMF:
    def test_sums_to_one(self):
        """PMF over enough values should sum to ~1."""
        lam = 2.5
        total = sum(poisson_pmf(k, lam) for k in range(30))
        assert float(total) == pytest.approx(1.0, abs=1e-10)

    def test_known_value(self):
        """P(X=0 | lam=1) = exp(-1) ≈ 0.3679."""
        p = poisson_pmf(0, 1.0)
        assert float(p) == pytest.approx(np.exp(-1.0))

    def test_mode_at_floor_lambda(self):
        """For integer lambda, mode is at lambda and lambda-1."""
        lam = 3.0
        probs = [float(poisson_pmf(k, lam)) for k in range(10)]
        assert probs[3] >= probs[2]
        assert probs[3] >= probs[4]


class TestPoissonLogPMF:
    def test_consistent_with_pmf(self):
        """log_pmf should equal log(pmf)."""
        for k in range(6):
            log_p = float(poisson_log_pmf(k, 2.0))
            p = float(poisson_pmf(k, 2.0))
            assert log_p == pytest.approx(np.log(p))

    def test_negative_values(self):
        """Log probabilities should be negative (probabilities < 1)."""
        for k in range(5):
            assert float(poisson_log_pmf(k, 1.5)) < 0
