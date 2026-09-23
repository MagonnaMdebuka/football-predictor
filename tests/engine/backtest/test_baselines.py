"""Tests for baseline predictors: uniform, base_rate, independent_poisson, bookmaker."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from services.engine.backtest.baselines import (
    BaseRateProbs,
    base_rate_from_training,
    base_rate_probabilities,
    bookmaker_probabilities,
    independent_poisson_probabilities,
    uniform_probabilities,
)


class TestUniformProbabilities:
    """Uniform 1/3 baseline."""

    def test_shape(self):
        h, d, a = uniform_probabilities(10)
        assert h.shape == (10,)
        assert d.shape == (10,)
        assert a.shape == (10,)

    def test_values(self):
        h, d, a = uniform_probabilities(5)
        np.testing.assert_allclose(h, 1 / 3)
        np.testing.assert_allclose(d, 1 / 3)
        np.testing.assert_allclose(a, 1 / 3)

    def test_sums_to_one(self):
        h, d, a = uniform_probabilities(3)
        totals = h + d + a
        np.testing.assert_allclose(totals, 1.0)

    def test_independent_arrays(self):
        """Modifying one array should not affect the others."""
        h, d, a = uniform_probabilities(2)
        h[0] = 0.5
        assert d[0] == pytest.approx(1 / 3)


class TestBaseRateFromTraining:
    """Compute base rates from training data."""

    def test_from_ftr_column(self):
        df = pd.DataFrame({"ftr": ["H", "H", "H", "D", "A"]})
        br = base_rate_from_training(df)
        assert br.home == pytest.approx(0.6)
        assert br.draw == pytest.approx(0.2)
        assert br.away == pytest.approx(0.2)

    def test_from_goals_columns(self):
        """Falls back to computing from goals when ftr is absent."""
        df = pd.DataFrame({
            "home_goals": [2, 1, 0, 3],
            "away_goals": [1, 1, 2, 0],
        })
        br = base_rate_from_training(df)
        assert br.home == pytest.approx(0.5)
        assert br.draw == pytest.approx(0.25)
        assert br.away == pytest.approx(0.25)

    def test_empty_returns_uniform(self):
        df = pd.DataFrame({"ftr": pd.Series([], dtype=str)})
        br = base_rate_from_training(df)
        assert br.home == pytest.approx(1 / 3)
        assert br.draw == pytest.approx(1 / 3)
        assert br.away == pytest.approx(1 / 3)

    def test_sums_to_one(self):
        df = pd.DataFrame({"ftr": ["H", "D", "A", "H", "H", "D", "A", "A", "A", "D"]})
        br = base_rate_from_training(df)
        assert br.home + br.draw + br.away == pytest.approx(1.0)

    def test_all_home_wins(self):
        df = pd.DataFrame({"ftr": ["H"] * 10})
        br = base_rate_from_training(df)
        assert br.home == pytest.approx(1.0)
        assert br.draw == pytest.approx(0.0)
        assert br.away == pytest.approx(0.0)


class TestBaseRateProbabilities:
    """Broadcast base rate to arrays."""

    def test_shape_and_values(self):
        br = BaseRateProbs(home=0.45, draw=0.27, away=0.28)
        h, d, a = base_rate_probabilities(br, 5)
        assert h.shape == (5,)
        np.testing.assert_allclose(h, 0.45)
        np.testing.assert_allclose(d, 0.27)
        np.testing.assert_allclose(a, 0.28)


class TestIndependentPoissonProbabilities:
    """Dixon-Coles with rho=0 (independent Poisson)."""

    @pytest.fixture
    def training_df(self) -> pd.DataFrame:
        """Small training set with 4 teams."""
        rng = np.random.default_rng(42)
        teams = ["A", "B", "C", "D"]
        rows = []
        for _ in range(5):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    rows.append({
                        "home_team": h,
                        "away_team": a,
                        "home_goals": int(rng.poisson(1.5)),
                        "away_goals": int(rng.poisson(1.2)),
                    })
        return pd.DataFrame(rows)

    def test_output_shape(self, training_df):
        pred_df = pd.DataFrame({
            "home_team": ["A", "B"],
            "away_team": ["B", "C"],
        })
        h, d, a = independent_poisson_probabilities(training_df, pred_df)
        assert h.shape == (2,)
        assert d.shape == (2,)
        assert a.shape == (2,)

    def test_sums_to_approximately_one(self, training_df):
        pred_df = pd.DataFrame({
            "home_team": ["A", "C"],
            "away_team": ["D", "B"],
        })
        h, d, a = independent_poisson_probabilities(training_df, pred_df)
        totals = h + d + a
        np.testing.assert_allclose(totals, 1.0, atol=0.01)

    def test_probabilities_in_range(self, training_df):
        pred_df = pd.DataFrame({
            "home_team": ["A"],
            "away_team": ["B"],
        })
        h, d, a = independent_poisson_probabilities(training_df, pred_df)
        assert 0 < h[0] < 1
        assert 0 < d[0] < 1
        assert 0 < a[0] < 1


class TestBookmakerProbabilities:
    """Bookmaker baseline from odds JSON."""

    def test_valid_odds(self):
        jsons = [
            json.dumps({"PSCH": 2.10, "PSCD": 3.30, "PSCA": 3.50}),
            json.dumps({"PSCH": 1.80, "PSCD": 3.50, "PSCA": 4.50}),
        ]
        h, d, a, excluded = bookmaker_probabilities(jsons)
        assert excluded == 0
        assert h is not None
        assert len(h) == 2
        # Probabilities should sum to ~1.0
        np.testing.assert_allclose(h + d + a, 1.0, atol=1e-10)

    def test_mixed_availability(self):
        jsons = [
            json.dumps({"PSCH": 2.10, "PSCD": 3.30, "PSCA": 3.50}),
            json.dumps({}),
        ]
        h, d, a, excluded = bookmaker_probabilities(jsons)
        assert excluded == 1
        assert np.isfinite(h[0])
        assert np.isnan(h[1])

    def test_all_missing(self):
        jsons = [json.dumps({})]
        h, d, a, excluded = bookmaker_probabilities(jsons)
        assert h is None
        assert excluded == 1
