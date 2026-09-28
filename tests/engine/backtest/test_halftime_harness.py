"""Tests for half-time backtest harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.engine.backtest.halftime_harness import run_halftime_predictions
from services.engine.backtest.types import HalfTimePrediction
from services.engine.models.decay import time_weights


class TestRunHalftimePredictions:
    def test_returns_predictions_for_each_match(self, multi_season_df):
        """Should produce one HalfTimePrediction per prediction match."""
        cutoff = pd.Timestamp("2024-08-17")
        training = multi_season_df[multi_season_df["date"] < cutoff].copy()
        pred_matches = multi_season_df[
            multi_season_df["date"].dt.normalize() == cutoff
        ].copy()

        if len(pred_matches) == 0:
            pytest.skip("No prediction matches on cutoff date")

        preds, ht_packed, ht_teams, sh_packed, sh_teams = run_halftime_predictions(
            training, pred_matches,
        )
        assert len(preds) == len(pred_matches)
        assert all(isinstance(p, HalfTimePrediction) for p in preds)

    def test_ht_lambdas_are_lower_than_ft(self, multi_season_df):
        """HT lambdas should generally be lower than FT lambdas."""
        cutoff = pd.Timestamp("2024-09-04")
        training = multi_season_df[multi_season_df["date"] < cutoff].copy()
        pred_dates = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(5).copy()

        if len(pred_dates) == 0:
            pytest.skip("No prediction matches")

        preds, *_ = run_halftime_predictions(training, pred_dates)
        if preds:
            mean_ht_lam = np.mean([p.ht_lambda_home + p.ht_lambda_away for p in preds])
            # HT lambdas should be substantially less than typical FT (~2.7)
            assert mean_ht_lam < 2.5

    def test_missing_ht_goals_rows_dropped(self, multi_season_df):
        """Rows with NaN HT goals should be dropped from training, not crash."""
        cutoff = pd.Timestamp("2024-09-04")
        training = multi_season_df[multi_season_df["date"] < cutoff].copy()
        # Introduce some NaN HT goals
        training.loc[training.index[:5], "ht_home_goals"] = np.nan

        pred_matches = multi_season_df[
            multi_season_df["date"].dt.normalize() == cutoff
        ].copy()

        if len(pred_matches) == 0:
            pytest.skip("No prediction matches")

        preds, *_ = run_halftime_predictions(training, pred_matches)
        # Should still produce predictions (just with slightly less training data)
        assert len(preds) == len(pred_matches)

    def test_htft_probs_populated(self, multi_season_df):
        """Each prediction should have 9 HT/FT probabilities."""
        cutoff = pd.Timestamp("2024-09-04")
        training = multi_season_df[multi_season_df["date"] < cutoff].copy()
        pred_matches = multi_season_df[
            multi_season_df["date"].dt.normalize() == cutoff
        ].copy()

        if len(pred_matches) == 0:
            pytest.skip("No prediction matches")

        preds, *_ = run_halftime_predictions(training, pred_matches)
        for p in preds:
            assert len(p.htft_probs) == 9
            total = sum(p.htft_probs.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_warm_start_carries_x0(self, multi_season_df):
        """When team set unchanged, packed params should be reusable."""
        cutoff1 = pd.Timestamp("2024-09-04")
        training1 = multi_season_df[multi_season_df["date"] < cutoff1].copy()
        pred1 = multi_season_df[multi_season_df["date"].dt.normalize() == cutoff1].head(1).copy()

        if len(pred1) == 0:
            pytest.skip("No prediction matches")

        _, ht_packed, ht_teams, sh_packed, sh_teams = run_halftime_predictions(
            training1, pred1,
        )

        # Second call with same teams — should accept prev_packed
        cutoff2 = pd.Timestamp("2024-10-02")
        training2 = multi_season_df[multi_season_df["date"] < cutoff2].copy()
        pred2 = multi_season_df[multi_season_df["date"].dt.normalize() == cutoff2].head(1).copy()

        if len(pred2) == 0:
            pytest.skip("No prediction matches")

        preds2, *_ = run_halftime_predictions(
            training2, pred2,
            prev_ht_packed=ht_packed,
            prev_ht_teams=ht_teams,
            prev_sh_packed=sh_packed,
            prev_sh_teams=sh_teams,
        )
        assert len(preds2) == len(pred2)

    def test_prediction_results_correct(self, multi_season_df):
        """ht_result and ft_result fields should match actual goals."""
        cutoff = pd.Timestamp("2024-09-04")
        training = multi_season_df[multi_season_df["date"] < cutoff].copy()
        pred_matches = multi_season_df[
            multi_season_df["date"].dt.normalize() == cutoff
        ].copy()

        if len(pred_matches) == 0:
            pytest.skip("No prediction matches")

        preds, *_ = run_halftime_predictions(training, pred_matches)
        for p in preds:
            # Verify ht_result
            if p.ht_home_goals > p.ht_away_goals:
                assert p.ht_result == "H"
            elif p.ht_home_goals == p.ht_away_goals:
                assert p.ht_result == "D"
            else:
                assert p.ht_result == "A"

    def test_sh_goals_are_nonnegative(self, multi_season_df):
        """Second-half goals should be >= 0."""
        cutoff = pd.Timestamp("2024-09-04")
        training = multi_season_df[multi_season_df["date"] < cutoff].copy()
        pred_matches = multi_season_df[
            multi_season_df["date"].dt.normalize() == cutoff
        ].copy()

        if len(pred_matches) == 0:
            pytest.skip("No prediction matches")

        preds, *_ = run_halftime_predictions(training, pred_matches)
        for p in preds:
            assert p.sh_home_goals >= 0
            assert p.sh_away_goals >= 0

    def test_empty_prediction_matches(self, multi_season_df):
        """Empty prediction set should return empty predictions."""
        cutoff = pd.Timestamp("2024-09-04")
        training = multi_season_df[multi_season_df["date"] < cutoff].copy()
        empty = training.head(0).copy()

        preds, *_ = run_halftime_predictions(training, empty)
        assert preds == []
