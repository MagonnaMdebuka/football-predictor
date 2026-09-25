"""Tests for NB2 count model negative log-likelihood."""

import numpy as np
import pytest

from services.engine.models.count_likelihood import neg_log_likelihood_count
from services.engine.models.count_params import CountModelParams, pack, vector_length


TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]


def _make_simple_vec(n_teams: int = 4, alpha: float = 0.3, n_refs: int = 0) -> np.ndarray:
    """Create a simple parameter vector."""
    vec_len = vector_length(n_teams, n_refs)
    vec = np.zeros(vec_len, dtype=np.float64)
    vec[0] = 0.15  # gamma
    vec[2 * n_teams - 1] = 2.0  # mu (log scale, exp(2)~7.4)
    vec[2 * n_teams] = alpha  # alpha
    return vec


def _make_match_data(n_matches: int = 12, rng_seed: int = 42):
    """Generate simple match data arrays."""
    rng = np.random.default_rng(rng_seed)
    home_idx = rng.integers(0, 4, size=n_matches).astype(np.int_)
    away_idx = rng.integers(0, 4, size=n_matches).astype(np.int_)
    # Ensure no team plays itself
    for i in range(n_matches):
        while away_idx[i] == home_idx[i]:
            away_idx[i] = rng.integers(0, 4)
    home_counts = rng.integers(0, 15, size=n_matches).astype(np.int_)
    away_counts = rng.integers(0, 15, size=n_matches).astype(np.int_)
    return home_idx, away_idx, home_counts, away_counts


class TestNegLogLikelihoodCount:
    def test_returns_finite_positive(self):
        """NLL should be finite and positive."""
        vec = _make_simple_vec()
        home_idx, away_idx, home_counts, away_counts = _make_match_data()
        nll = neg_log_likelihood_count(
            vec, TEAMS, home_idx, away_idx, home_counts, away_counts,
        )
        assert np.isfinite(nll)
        assert nll > 0

    def test_weights_affect_result(self):
        """Doubling weights should roughly double NLL."""
        vec = _make_simple_vec()
        home_idx, away_idx, home_counts, away_counts = _make_match_data()

        nll_uniform = neg_log_likelihood_count(
            vec, TEAMS, home_idx, away_idx, home_counts, away_counts,
        )
        weights_double = 2.0 * np.ones(len(home_idx), dtype=np.float64)
        nll_double = neg_log_likelihood_count(
            vec, TEAMS, home_idx, away_idx, home_counts, away_counts,
            weights=weights_double,
        )
        assert nll_double == pytest.approx(2.0 * nll_uniform, rel=1e-6)

    def test_zero_weight_ignores_match(self):
        """Match with weight=0 should not contribute to NLL."""
        vec = _make_simple_vec()
        home_idx, away_idx, home_counts, away_counts = _make_match_data(n_matches=5)

        weights = np.ones(5, dtype=np.float64)
        nll_all = neg_log_likelihood_count(
            vec, TEAMS, home_idx, away_idx, home_counts, away_counts,
            weights=weights,
        )
        weights_zero_last = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        nll_skip_last = neg_log_likelihood_count(
            vec, TEAMS, home_idx, away_idx, home_counts, away_counts,
            weights=weights_zero_last,
        )
        # Should differ since one match is removed
        nll_first_four = neg_log_likelihood_count(
            vec, TEAMS, home_idx[:4], away_idx[:4],
            home_counts[:4], away_counts[:4],
        )
        assert nll_skip_last == pytest.approx(nll_first_four, rel=1e-6)

    def test_with_referees(self):
        """NLL should be computable with referee effects."""
        refs = ["RefA", "RefB", "RefC"]
        vec = _make_simple_vec(n_refs=3)
        home_idx, away_idx, home_counts, away_counts = _make_match_data()
        rng = np.random.default_rng(99)
        ref_idx = rng.integers(0, 3, size=len(home_idx)).astype(np.int_)

        nll = neg_log_likelihood_count(
            vec, TEAMS, home_idx, away_idx, home_counts, away_counts,
            referees=refs, ref_idx=ref_idx,
        )
        assert np.isfinite(nll)
        assert nll > 0

    def test_alpha_near_zero_finite(self):
        """NLL should be finite even with alpha near zero (Poisson fallback)."""
        vec = _make_simple_vec(alpha=1e-12)
        home_idx, away_idx, home_counts, away_counts = _make_match_data()
        nll = neg_log_likelihood_count(
            vec, TEAMS, home_idx, away_idx, home_counts, away_counts,
        )
        assert np.isfinite(nll)
        assert nll > 0

    def test_better_params_lower_nll(self):
        """Parameters matching the data should have lower NLL than random."""
        rng = np.random.default_rng(42)
        n_matches = 50
        home_idx = rng.integers(0, 4, size=n_matches).astype(np.int_)
        away_idx = np.array([(h + 1) % 4 for h in home_idx], dtype=np.int_)
        # Generate from known mu (all counts ~ exp(2) ~ 7.4)
        home_counts = rng.poisson(7.4, size=n_matches).astype(np.int_)
        away_counts = rng.poisson(6.0, size=n_matches).astype(np.int_)

        # Good params: mu that matches the data generating process
        vec_good = _make_simple_vec(alpha=0.01)
        vec_good[2 * 4 - 1] = np.log(7.0)  # mu close to true

        # Bad params: very wrong mu
        vec_bad = _make_simple_vec(alpha=0.01)
        vec_bad[2 * 4 - 1] = np.log(1.0)  # mu too low

        nll_good = neg_log_likelihood_count(
            vec_good, TEAMS, home_idx, away_idx, home_counts, away_counts,
        )
        nll_bad = neg_log_likelihood_count(
            vec_bad, TEAMS, home_idx, away_idx, home_counts, away_counts,
        )
        assert nll_good < nll_bad
