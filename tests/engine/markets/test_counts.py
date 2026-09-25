"""Tests for count-derived markets (corners and booking points)."""

import numpy as np
import pytest

from services.engine.markets.counts import (
    BOOKING_MATCH_LINES,
    BOOKING_TEAM_LINES,
    CORNER_MATCH_LINES,
    CORNER_TEAM_LINES,
    CountMarkets,
    compound_booking_to_markets,
    count_to_markets,
)


class TestCountToMarkets:
    """Over/under markets from NB2 convolution."""

    def _corner_markets(self, mu_home=5.0, mu_away=4.5, alpha=0.15) -> CountMarkets:
        return count_to_markets(
            mu_home, mu_away, alpha,
            match_lines=CORNER_MATCH_LINES,
            team_lines=CORNER_TEAM_LINES,
        )

    def _booking_markets(self, mu_home=25.0, mu_away=22.0, alpha=0.25) -> CountMarkets:
        return count_to_markets(
            mu_home, mu_away, alpha,
            match_lines=BOOKING_MATCH_LINES,
            team_lines=BOOKING_TEAM_LINES,
        )

    def test_over_plus_under_equals_one_match(self):
        """Over + under should sum to 1 for each match line."""
        markets = self._corner_markets()
        for ou in markets.match_over_under:
            assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=0.001)

    def test_over_plus_under_equals_one_home(self):
        """Over + under should sum to 1 for each home team line."""
        markets = self._corner_markets()
        for ou in markets.home_over_under:
            assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=0.001)

    def test_over_plus_under_equals_one_away(self):
        """Over + under should sum to 1 for each away team line."""
        markets = self._corner_markets()
        for ou in markets.away_over_under:
            assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=0.001)

    def test_monotonicity_match_over(self):
        """P(over) should decrease as line increases."""
        markets = self._corner_markets()
        overs = [ou["over"] for ou in markets.match_over_under]
        for i in range(len(overs) - 1):
            assert overs[i] >= overs[i + 1] - 1e-6

    def test_monotonicity_match_under(self):
        """P(under) should increase as line increases."""
        markets = self._corner_markets()
        unders = [ou["under"] for ou in markets.match_over_under]
        for i in range(len(unders) - 1):
            assert unders[i] <= unders[i + 1] + 1e-6

    def test_monotonicity_home_over(self):
        markets = self._corner_markets()
        overs = [ou["over"] for ou in markets.home_over_under]
        for i in range(len(overs) - 1):
            assert overs[i] >= overs[i + 1] - 1e-6

    def test_monotonicity_away_over(self):
        markets = self._corner_markets()
        overs = [ou["over"] for ou in markets.away_over_under]
        for i in range(len(overs) - 1):
            assert overs[i] >= overs[i + 1] - 1e-6

    def test_booking_point_markets(self):
        """Booking point markets should produce valid results."""
        markets = self._booking_markets()
        assert len(markets.match_over_under) == len(BOOKING_MATCH_LINES)
        assert len(markets.home_over_under) == len(BOOKING_TEAM_LINES)
        for ou in markets.match_over_under:
            assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=0.001)

    def test_alpha_zero_matches_poisson(self):
        """With alpha~0, results should match Poisson convolution."""
        markets_nb = count_to_markets(
            5.0, 4.5, alpha=1e-12,
            match_lines=CORNER_MATCH_LINES,
            team_lines=CORNER_TEAM_LINES,
        )
        # Compare against Poisson-based calculation
        from scipy.stats import poisson
        pmf_home = np.array([poisson.pmf(k, 5.0) for k in range(81)])
        pmf_away = np.array([poisson.pmf(k, 4.5) for k in range(81)])
        pmf_total = np.convolve(pmf_home, pmf_away)
        cdf_total = np.cumsum(pmf_total)

        for ou in markets_nb.match_over_under:
            k = int(ou["line"])
            expected_under = float(cdf_total[k])
            assert ou["under"] == pytest.approx(expected_under, abs=0.002)

    def test_higher_mu_increases_over(self):
        """Higher expected count should increase P(over) for all lines."""
        markets_low = count_to_markets(
            3.0, 3.0, 0.15,
            match_lines=CORNER_MATCH_LINES,
            team_lines=CORNER_TEAM_LINES,
        )
        markets_high = count_to_markets(
            8.0, 7.0, 0.15,
            match_lines=CORNER_MATCH_LINES,
            team_lines=CORNER_TEAM_LINES,
        )
        for low_ou, high_ou in zip(
            markets_low.match_over_under, markets_high.match_over_under,
        ):
            assert high_ou["over"] > low_ou["over"]

    def test_returns_correct_line_counts(self):
        """Markets should have the right number of entries per line list."""
        markets = self._corner_markets()
        assert len(markets.match_over_under) == len(CORNER_MATCH_LINES)
        assert len(markets.home_over_under) == len(CORNER_TEAM_LINES)
        assert len(markets.away_over_under) == len(CORNER_TEAM_LINES)

    def test_probabilities_in_valid_range(self):
        """All probabilities should be in [0, 1]."""
        markets = self._corner_markets()
        for ou in markets.match_over_under + markets.home_over_under + markets.away_over_under:
            assert 0.0 <= ou["over"] <= 1.0
            assert 0.0 <= ou["under"] <= 1.0


class TestCompoundBookingToMarkets:
    """Compound booking-point markets from NB2 yellows + Poisson reds."""

    def _default_markets(
        self,
        mu_y_h: float = 3.0,
        mu_y_a: float = 2.5,
        alpha_y: float = 0.15,
        mu_r_h: float = 0.1,
        mu_r_a: float = 0.08,
    ) -> CountMarkets:
        return compound_booking_to_markets(
            mu_yellow_home=mu_y_h,
            mu_yellow_away=mu_y_a,
            alpha_yellow=alpha_y,
            mu_red_home=mu_r_h,
            mu_red_away=mu_r_a,
            match_lines=BOOKING_MATCH_LINES,
            team_lines=BOOKING_TEAM_LINES,
        )

    def test_compound_over_under_sums_to_one(self):
        """Over + under should sum to 1 for each match line."""
        markets = self._default_markets()
        for ou in markets.match_over_under:
            assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=0.001)
        for ou in markets.home_over_under:
            assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=0.001)
        for ou in markets.away_over_under:
            assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=0.001)

    def test_compound_monotonicity(self):
        """P(over) should decrease as line increases."""
        markets = self._default_markets()
        overs = [ou["over"] for ou in markets.match_over_under]
        for i in range(len(overs) - 1):
            assert overs[i] >= overs[i + 1] - 1e-6

    def test_compound_probabilities_valid_range(self):
        """All probabilities should be in [0, 1]."""
        markets = self._default_markets()
        for ou in (
            markets.match_over_under
            + markets.home_over_under
            + markets.away_over_under
        ):
            assert 0.0 <= ou["over"] <= 1.0
            assert 0.0 <= ou["under"] <= 1.0

    def test_compound_higher_mu_yellow_increases_over(self):
        """Higher yellow rate should increase P(over) for all match lines."""
        markets_low = self._default_markets(mu_y_h=2.0, mu_y_a=2.0)
        markets_high = self._default_markets(mu_y_h=5.0, mu_y_a=4.5)
        for low_ou, high_ou in zip(
            markets_low.match_over_under, markets_high.match_over_under,
        ):
            assert high_ou["over"] >= low_ou["over"] - 1e-6

    def test_compound_returns_correct_line_counts(self):
        """Markets should have the right number of entries."""
        markets = self._default_markets()
        assert len(markets.match_over_under) == len(BOOKING_MATCH_LINES)
        assert len(markets.home_over_under) == len(BOOKING_TEAM_LINES)
        assert len(markets.away_over_under) == len(BOOKING_TEAM_LINES)
