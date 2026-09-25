"""Tests for negative binomial count model functions."""

import numpy as np
import pytest

from services.engine.models.negbin import (
    count_expectancy,
    negbin_cdf,
    negbin_log_pmf,
    negbin_pmf,
)
from services.engine.models.poisson import poisson_pmf


class TestCountExpectancy:
    def test_home_advantage_increases_mu(self):
        """Positive gamma should give mu_home > mu_away for equal teams."""
        mu_h, mu_a = count_expectancy(
            attack_h=0.0, defence_a=0.0,
            attack_a=0.0, defence_h=0.0,
            mu=0.0, gamma=0.3,
        )
        assert mu_h[0] > mu_a[0]

    def test_symmetric_when_no_home_advantage(self):
        """With gamma=0 and identical params, mu_home == mu_away."""
        mu_h, mu_a = count_expectancy(
            attack_h=0.1, defence_a=-0.1,
            attack_a=0.1, defence_h=-0.1,
            mu=0.2, gamma=0.0,
        )
        assert mu_h[0] == pytest.approx(mu_a[0])

    def test_referee_effect_increases_mu(self):
        """Positive referee effect should increase the expected count."""
        mu_h_no_ref, _ = count_expectancy(
            attack_h=0.0, defence_a=0.0,
            attack_a=0.0, defence_h=0.0,
            mu=0.2, gamma=0.1,
        )
        mu_h_ref, _ = count_expectancy(
            attack_h=0.0, defence_a=0.0,
            attack_a=0.0, defence_h=0.0,
            mu=0.2, gamma=0.1,
            ref_effect_h=0.3, ref_effect_a=0.3,
        )
        assert mu_h_ref[0] > mu_h_no_ref[0]

    def test_vectorised_input(self):
        attack_h = np.array([0.1, 0.2, 0.3])
        defence_a = np.array([0.0, -0.1, 0.1])
        attack_a = np.array([0.0, 0.1, -0.1])
        defence_h = np.array([0.1, 0.0, 0.2])
        mu_h, mu_a = count_expectancy(
            attack_h, defence_a, attack_a, defence_h, mu=0.2, gamma=0.1,
        )
        assert mu_h.shape == (3,)
        assert mu_a.shape == (3,)

    def test_positive_mus(self):
        """Expected counts must always be positive."""
        mu_h, mu_a = count_expectancy(
            attack_h=-5.0, defence_a=-5.0,
            attack_a=-5.0, defence_h=-5.0,
            mu=-5.0, gamma=-5.0,
        )
        assert mu_h[0] > 0
        assert mu_a[0] > 0


class TestNegbinPMF:
    def test_sums_to_approximately_one(self):
        """PMF over enough values should sum to ~1."""
        mu_val = 5.0
        alpha = 0.5
        total = sum(negbin_pmf(k, mu_val, alpha).item() for k in range(50))
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_known_value_poisson_limit(self):
        """When alpha is near zero, NB should match Poisson."""
        mu_val = 2.5
        for k in range(8):
            nb_p = negbin_pmf(k, mu_val, alpha=0.0).item()
            poi_p = float(poisson_pmf(k, mu_val))
            assert nb_p == pytest.approx(poi_p, rel=1e-6)

    def test_overdispersion_spreads_mass(self):
        """Higher alpha should spread mass to higher counts (fatter tail)."""
        mu_val = 5.0
        p_alpha_small = negbin_pmf(0, mu_val, alpha=0.01).item()
        p_alpha_large = negbin_pmf(0, mu_val, alpha=1.0).item()
        assert p_alpha_large > p_alpha_small

    def test_vectorised_mu(self):
        """Should handle arrays of mu values."""
        mu_vals = np.array([2.0, 5.0, 10.0])
        result = negbin_pmf(3, mu_vals, alpha=0.5)
        assert result.shape == (3,)
        assert all(result > 0)

    def test_non_negative(self):
        """PMF values must be non-negative."""
        for k in range(15):
            p = negbin_pmf(k, 5.0, alpha=0.5).item()
            assert p >= 0


class TestNegbinLogPMF:
    def test_consistent_with_pmf(self):
        """log_pmf should equal log(pmf)."""
        mu_val = 5.0
        alpha = 0.3
        for k in range(8):
            log_p = negbin_log_pmf(k, mu_val, alpha).item()
            p = negbin_pmf(k, mu_val, alpha).item()
            assert log_p == pytest.approx(np.log(p), abs=1e-10)

    def test_negative_values(self):
        """Log probabilities should be negative (probabilities < 1)."""
        for k in range(5):
            assert negbin_log_pmf(k, 5.0, 0.3).item() < 0

    def test_alpha_zero_matches_poisson(self):
        """With alpha=0, log_pmf should match Poisson logpmf."""
        from services.engine.models.poisson import poisson_log_pmf

        mu_val = 3.0
        for k in range(6):
            nb_log_p = negbin_log_pmf(k, mu_val, alpha=0.0).item()
            poi_log_p = float(poisson_log_pmf(k, mu_val))
            assert nb_log_p == pytest.approx(poi_log_p, rel=1e-6)


class TestNegbinCDF:
    def test_monotonically_increasing(self):
        """CDF should be monotonically non-decreasing."""
        mu_val = 5.0
        alpha = 0.5
        prev = 0.0
        for k in range(20):
            cdf_val = negbin_cdf(k, mu_val, alpha).item()
            assert cdf_val >= prev - 1e-12
            prev = cdf_val

    def test_approaches_one(self):
        """CDF at large k should approach 1."""
        cdf_val = negbin_cdf(100, 5.0, alpha=0.5).item()
        assert cdf_val == pytest.approx(1.0, abs=1e-8)

    def test_cdf_at_zero(self):
        """CDF(0) should equal PMF(0)."""
        mu_val = 5.0
        alpha = 0.3
        cdf_0 = negbin_cdf(0, mu_val, alpha).item()
        pmf_0 = negbin_pmf(0, mu_val, alpha).item()
        assert cdf_0 == pytest.approx(pmf_0)

    def test_alpha_zero_matches_poisson(self):
        """With alpha=0, CDF should match Poisson CDF."""
        from scipy.stats import poisson as poisson_dist

        mu_val = 3.0
        for k in range(8):
            nb_cdf = negbin_cdf(k, mu_val, alpha=0.0).item()
            poi_cdf = float(poisson_dist.cdf(k, mu_val))
            assert nb_cdf == pytest.approx(poi_cdf, rel=1e-6)

    def test_vectorised_mu(self):
        """Should handle arrays of mu values."""
        mu_vals = np.array([2.0, 5.0, 10.0])
        result = negbin_cdf(5, mu_vals, alpha=0.5)
        assert result.shape == (3,)
