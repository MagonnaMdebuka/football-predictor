"""Tests for Dixon-Coles tau correction."""

import numpy as np
import pytest

from services.engine.models.tau import tau


class TestTauCases:
    """Verify all five tau cases from Dixon & Coles (1997)."""

    def test_zero_zero(self):
        result = tau(0, 0, lambda_home=1.5, lambda_away=1.2, rho=-0.1)
        expected = 1.0 - 1.5 * 1.2 * (-0.1)
        assert float(result[0]) == pytest.approx(expected)

    def test_zero_one(self):
        result = tau(0, 1, lambda_home=1.5, lambda_away=1.2, rho=-0.1)
        expected = 1.0 + 1.5 * (-0.1)
        assert float(result[0]) == pytest.approx(expected)

    def test_one_zero(self):
        result = tau(1, 0, lambda_home=1.5, lambda_away=1.2, rho=-0.1)
        expected = 1.0 + 1.2 * (-0.1)
        assert float(result[0]) == pytest.approx(expected)

    def test_one_one(self):
        result = tau(1, 1, lambda_home=1.5, lambda_away=1.2, rho=-0.1)
        expected = 1.0 - (-0.1)
        assert float(result[0]) == pytest.approx(expected)

    def test_other_scores_return_one(self):
        for h, a in [(2, 0), (0, 2), (3, 1), (2, 2), (5, 3)]:
            result = tau(h, a, lambda_home=1.5, lambda_away=1.2, rho=-0.1)
            assert float(result[0]) == pytest.approx(1.0)


class TestRhoZero:
    """When rho=0, tau should always return 1.0 (independent Poisson)."""

    def test_all_low_scores_return_one(self):
        for h, a in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            result = tau(h, a, lambda_home=2.0, lambda_away=1.0, rho=0.0)
            assert float(result[0]) == pytest.approx(1.0)


class TestVectorised:
    def test_mixed_scores(self):
        home = np.array([0, 0, 1, 1, 2])
        away = np.array([0, 1, 0, 1, 3])
        lam_h = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        lam_a = np.array([1.2, 1.2, 1.2, 1.2, 1.2])
        result = tau(home, away, lam_h, lam_a, rho=-0.1)
        assert result.shape == (5,)
        # Last element (2,3) should be 1.0
        assert float(result[4]) == pytest.approx(1.0)
        # First element (0,0) should match scalar test
        assert float(result[0]) == pytest.approx(1.0 - 1.5 * 1.2 * (-0.1))
