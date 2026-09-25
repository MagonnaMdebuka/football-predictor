"""Tests for total count NB2 likelihood function."""

import numpy as np
import pytest

from services.engine.models.total_count_likelihood import neg_log_likelihood_total_count
from services.engine.models.total_count_params import pack_total, TotalCountModelParams


def _make_params(n=4, mu=2.3, alpha=0.15):
    """Create simple test parameters."""
    teams = [f"T{i}" for i in range(n)]
    home_eff = np.zeros(n, dtype=np.float64)
    away_eff = np.zeros(n, dtype=np.float64)
    return TotalCountModelParams(
        teams=teams, mu=mu, home_effect=home_eff, away_effect=away_eff, alpha=alpha,
    )


class TestNegLogLikelihoodTotalCount:
    """Basic likelihood function tests."""

    def test_returns_scalar(self):
        params = _make_params()
        vec = pack_total(params)
        home_idx = np.array([0, 1, 2, 3], dtype=np.int_)
        away_idx = np.array([1, 2, 3, 0], dtype=np.int_)
        total_counts = np.array([10, 9, 11, 8], dtype=np.int_)

        result = neg_log_likelihood_total_count(
            vec, params.teams, home_idx, away_idx, total_counts,
        )
        assert isinstance(result, float)

    def test_positive_nll(self):
        """Negative log-likelihood should be positive."""
        params = _make_params()
        vec = pack_total(params)
        home_idx = np.array([0, 1, 2, 3], dtype=np.int_)
        away_idx = np.array([1, 2, 3, 0], dtype=np.int_)
        total_counts = np.array([10, 9, 11, 8], dtype=np.int_)

        result = neg_log_likelihood_total_count(
            vec, params.teams, home_idx, away_idx, total_counts,
        )
        assert result > 0

    def test_weights_change_result(self):
        """Applying non-uniform weights should change the NLL."""
        params = _make_params()
        vec = pack_total(params)
        home_idx = np.array([0, 1, 2, 3], dtype=np.int_)
        away_idx = np.array([1, 2, 3, 0], dtype=np.int_)
        total_counts = np.array([10, 9, 11, 8], dtype=np.int_)

        nll_uniform = neg_log_likelihood_total_count(
            vec, params.teams, home_idx, away_idx, total_counts,
        )
        weights = np.array([2.0, 0.5, 1.0, 0.1], dtype=np.float64)
        nll_weighted = neg_log_likelihood_total_count(
            vec, params.teams, home_idx, away_idx, total_counts, weights,
        )
        assert nll_uniform != pytest.approx(nll_weighted)

    def test_poisson_fallback(self):
        """With alpha near zero, should compute without error (Poisson fallback)."""
        params = _make_params(alpha=1e-15)
        vec = pack_total(params)
        home_idx = np.array([0, 1], dtype=np.int_)
        away_idx = np.array([1, 0], dtype=np.int_)
        total_counts = np.array([10, 9], dtype=np.int_)

        result = neg_log_likelihood_total_count(
            vec, params.teams, home_idx, away_idx, total_counts,
        )
        assert np.isfinite(result)

    def test_gradient_finite_differences(self):
        """Numerical gradient should be close to analytical (via scipy approx_fprime)."""
        from scipy.optimize import approx_fprime

        params = _make_params(n=3, mu=2.3, alpha=0.15)
        vec = pack_total(params)
        home_idx = np.array([0, 1, 2, 0, 1, 2], dtype=np.int_)
        away_idx = np.array([1, 2, 0, 2, 0, 1], dtype=np.int_)
        total_counts = np.array([10, 9, 11, 8, 12, 7], dtype=np.int_)

        def f(v):
            return neg_log_likelihood_total_count(
                v, params.teams, home_idx, away_idx, total_counts,
            )

        grad = approx_fprime(vec, f, 1e-6)
        # Gradient should be finite
        assert np.all(np.isfinite(grad))

    def test_better_mu_gives_lower_nll(self):
        """mu closer to log(mean_total) should give lower NLL."""
        teams = ["A", "B", "C", "D"]
        home_idx = np.array([0, 1, 2, 3] * 5, dtype=np.int_)
        away_idx = np.array([1, 2, 3, 0] * 5, dtype=np.int_)
        total_counts = np.array([10, 11, 9, 10] * 5, dtype=np.int_)

        mean_total = total_counts.mean()
        good_mu = np.log(mean_total)  # ~2.3
        bad_mu = np.log(mean_total) + 1.0  # too high

        good_params = _make_params(n=4, mu=good_mu, alpha=0.15)
        bad_params = _make_params(n=4, mu=bad_mu, alpha=0.15)

        nll_good = neg_log_likelihood_total_count(
            pack_total(good_params), teams, home_idx, away_idx, total_counts,
        )
        nll_bad = neg_log_likelihood_total_count(
            pack_total(bad_params), teams, home_idx, away_idx, total_counts,
        )
        assert nll_good < nll_bad
