"""Tests for time-decay weights."""

from datetime import date

import numpy as np
import pytest

from services.engine.models.decay import (
    half_life_days,
    optimise_xi,
    time_weights,
    xi_from_half_life,
)


class TestTimeWeights:
    def test_weight_one_at_reference_date(self):
        """Match on the reference date should have weight 1.0."""
        ref = date(2024, 6, 1)
        dates = np.array([np.datetime64("2024-06-01")])
        w = time_weights(dates, ref, xi=0.005)
        assert float(w[0]) == pytest.approx(1.0)

    def test_weight_decreases_with_age(self):
        """Older matches should have lower weight."""
        ref = date(2024, 6, 1)
        dates = np.array([
            np.datetime64("2024-06-01"),
            np.datetime64("2024-05-01"),
            np.datetime64("2024-01-01"),
        ])
        w = time_weights(dates, ref, xi=0.005)
        assert w[0] > w[1] > w[2]

    def test_all_weights_positive(self):
        """Weights should always be positive (exp never returns zero)."""
        ref = date(2024, 6, 1)
        dates = np.array([np.datetime64("2020-01-01"), np.datetime64("2015-06-01")])
        w = time_weights(dates, ref, xi=0.01)
        assert np.all(w > 0)

    def test_zero_xi_gives_uniform_weights(self):
        """With xi=0, all matches get equal weight of 1.0."""
        ref = date(2024, 6, 1)
        dates = np.array([
            np.datetime64("2024-06-01"),
            np.datetime64("2023-01-01"),
            np.datetime64("2020-06-01"),
        ])
        w = time_weights(dates, ref, xi=0.0)
        np.testing.assert_allclose(w, 1.0)

    def test_accepts_python_dates(self):
        """Should accept a list of Python date objects."""
        ref = date(2024, 6, 1)
        dates = [date(2024, 5, 1), date(2024, 4, 1)]
        w = time_weights(dates, ref, xi=0.005)
        assert w.shape == (2,)
        assert np.all(w > 0)


class TestHalfLife:
    def test_half_life_property(self):
        """Weight at exactly the half-life should be 0.5."""
        xi = 0.005
        hl = half_life_days(xi)
        ref = date(2024, 6, 1)
        match_date = np.datetime64("2024-06-01") - np.timedelta64(round(hl), "D")
        w = time_weights(np.array([match_date]), ref, xi=xi)
        assert float(w[0]) == pytest.approx(0.5, abs=0.02)

    def test_round_trip(self):
        """xi_from_half_life(half_life_days(xi)) should return xi."""
        xi = 0.007
        hl = half_life_days(xi)
        assert xi_from_half_life(hl) == pytest.approx(xi)

    def test_shorter_half_life_means_larger_xi(self):
        """Faster decay (shorter half-life) requires larger xi."""
        xi_slow = xi_from_half_life(180)
        xi_fast = xi_from_half_life(30)
        assert xi_fast > xi_slow


class TestOptimiseXi:
    def test_returns_best_xi(self, synthetic_df):
        """optimise_xi should return a positive xi and scores for each candidate."""
        candidates = np.array([0.001, 0.005, 0.010])
        best_xi, scores = optimise_xi(synthetic_df, xi_candidates=candidates)
        assert best_xi in [0.001, 0.005, 0.010]
        assert len(scores) == 3
        # Best xi should have the lowest NLL
        best_nll = min(s[1] for s in scores)
        assert dict(scores)[best_xi] == pytest.approx(best_nll)

    def test_all_nlls_finite(self, synthetic_df):
        """All NLL values should be finite."""
        candidates = np.array([0.002, 0.005])
        _, scores = optimise_xi(synthetic_df, xi_candidates=candidates)
        for _, nll in scores:
            assert np.isfinite(nll)
