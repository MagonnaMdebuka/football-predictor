"""Tests for count model backtest harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.engine.backtest.count_harness import (
    run_compound_card_predictions,
    run_count_predictions,
)


class TestRunCountPredictions:
    """Integration tests for count model fit + predict pipeline."""

    def test_corners_produces_predictions(self, multi_season_df):
        """Corner model produces predictions for held-out matches."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(5)

        preds, packed, teams = run_count_predictions(
            training, held_out, model_type="corners",
        )
        assert len(preds) == len(held_out)
        assert all(p.model_type == "corners" for p in preds)
        assert len(packed) > 0
        assert len(teams) > 0

    def test_cards_produces_predictions(self, multi_season_df):
        """Card model produces predictions for held-out matches."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(5)

        preds, packed, teams = run_count_predictions(
            training, held_out, model_type="cards",
            include_referees=True, min_referee_matches=5,
        )
        assert len(preds) == len(held_out)
        assert all(p.model_type == "cards" for p in preds)

    def test_predictions_have_markets(self, multi_season_df):
        """Each prediction has match and team-level O/U entries."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        preds, _, _ = run_count_predictions(
            training, held_out, model_type="corners",
        )
        for p in preds:
            assert len(p.match_over_under) > 0
            assert len(p.home_over_under) > 0
            assert len(p.away_over_under) > 0
            # Over + under should sum to ~1
            for ou in p.match_over_under:
                assert abs(ou["over"] + ou["under"] - 1.0) < 0.01

    def test_positive_mu_values(self, multi_season_df):
        """Predicted mu values must be positive."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        preds, _, _ = run_count_predictions(
            training, held_out, model_type="corners",
        )
        for p in preds:
            assert p.mu_home > 0
            assert p.mu_away > 0

    def test_warm_start_accepted(self, multi_season_df):
        """Previous packed params enable warm-starting."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        # First fit
        preds1, packed1, teams1 = run_count_predictions(
            training, held_out, model_type="corners",
        )
        # Second fit with warm-start
        preds2, packed2, teams2 = run_count_predictions(
            training, held_out, model_type="corners",
            prev_packed=packed1, prev_teams=teams1,
        )
        assert len(preds2) == len(preds1)

    def test_missing_columns_returns_empty(self):
        """Gracefully returns empty list if target columns are missing."""
        training = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 6),
            "home_team": ["A", "B", "C", "A", "B", "C"],
            "away_team": ["B", "C", "A", "C", "A", "B"],
            "home_goals": [1, 2, 0, 1, 3, 1],
            "away_goals": [0, 1, 1, 2, 0, 0],
        })
        held_out = training.head(2)
        preds, packed, teams = run_count_predictions(
            training, held_out, model_type="corners",
        )
        assert preds == []

    def test_insufficient_training_returns_empty(self):
        """Returns empty list if training data has fewer than 10 matches."""
        training = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "home_team": ["A", "B", "A", "B"],
            "away_team": ["B", "A", "B", "A"],
            "home_corners": [5, 6, 4, 7],
            "away_corners": [3, 4, 5, 2],
        })
        held_out = training.head(2)
        preds, _, _ = run_count_predictions(
            training, held_out, model_type="corners",
        )
        assert preds == []

    def test_cards_referee_in_prediction(self, multi_season_df):
        """Card predictions include referee name when available."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        preds, _, _ = run_count_predictions(
            training, held_out, model_type="cards",
            include_referees=True, min_referee_matches=5,
        )
        for p in preds:
            assert p.referee is not None

    def test_n_training_matches_positive(self, multi_season_df):
        """Training match count recorded in predictions."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        preds, _, _ = run_count_predictions(
            training, held_out, model_type="corners",
        )
        for p in preds:
            assert p.n_training_matches > 0

    def test_alpha_positive(self, multi_season_df):
        """Alpha (overdispersion) parameter must be positive."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        preds, _, _ = run_count_predictions(
            training, held_out, model_type="corners",
        )
        for p in preds:
            assert p.alpha > 0


class TestRunCompoundCardPredictions:
    """Integration tests for compound yellow/red card pipeline."""

    def test_compound_cards_produces_predictions(self, multi_season_df):
        """Compound card model produces predictions for held-out matches."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(5)

        preds, y_packed, y_teams, r_packed, r_teams = run_compound_card_predictions(
            training, held_out, min_referee_matches=5,
        )
        assert len(preds) == len(held_out)
        assert all(p.model_type == "cards" for p in preds)

    def test_compound_cards_has_yellow_red_fields(self, multi_season_df):
        """Compound predictions should have yellow/red fields populated."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        preds, *_ = run_compound_card_predictions(
            training, held_out, min_referee_matches=5,
        )
        for p in preds:
            assert p.mu_yellow_home is not None
            assert p.mu_yellow_away is not None
            assert p.alpha_yellow is not None
            assert p.mu_red_home is not None
            assert p.mu_red_away is not None
            assert p.mu_yellow_home > 0
            assert p.mu_red_home > 0

    def test_compound_mu_equals_weighted_sum(self, multi_season_df):
        """mu_home should equal 10*mu_yellow_home + 25*mu_red_home."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        preds, *_ = run_compound_card_predictions(
            training, held_out, min_referee_matches=5,
        )
        for p in preds:
            expected_home = 10.0 * p.mu_yellow_home + 25.0 * p.mu_red_home
            expected_away = 10.0 * p.mu_yellow_away + 25.0 * p.mu_red_away
            assert p.mu_home == pytest.approx(expected_home, abs=0.01)
            assert p.mu_away == pytest.approx(expected_away, abs=0.01)

    def test_compound_cards_warm_start(self, multi_season_df):
        """Warm-start with previous packed params should work."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        # First fit
        preds1, yp1, yt1, rp1, rt1 = run_compound_card_predictions(
            training, held_out, min_referee_matches=5,
        )
        # Second fit with warm-start
        preds2, yp2, yt2, rp2, rt2 = run_compound_card_predictions(
            training, held_out, min_referee_matches=5,
            prev_yellow_packed=yp1, prev_yellow_teams=yt1,
            prev_red_packed=rp1, prev_red_teams=rt1,
        )
        assert len(preds2) == len(preds1)

    def test_compound_missing_columns_returns_empty(self):
        """Returns empty list if yellow/red columns are missing."""
        training = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 6),
            "home_team": ["A", "B", "C", "A", "B", "C"],
            "away_team": ["B", "C", "A", "C", "A", "B"],
            "home_goals": [1, 2, 0, 1, 3, 1],
            "away_goals": [0, 1, 1, 2, 0, 0],
        })
        held_out = training.head(2)
        preds, *_ = run_compound_card_predictions(training, held_out)
        assert preds == []

    def test_compound_has_markets(self, multi_season_df):
        """Each compound prediction has match and team-level O/U entries."""
        cutoff = pd.Timestamp("2024-08-01")
        training = multi_season_df[multi_season_df["date"] < cutoff]
        held_out = multi_season_df[
            (multi_season_df["date"] >= cutoff)
            & (multi_season_df["season"] == "2024-25")
        ].head(3)

        preds, *_ = run_compound_card_predictions(
            training, held_out, min_referee_matches=5,
        )
        for p in preds:
            assert len(p.match_over_under) > 0
            assert len(p.home_over_under) > 0
            for ou in p.match_over_under:
                assert abs(ou["over"] + ou["under"] - 1.0) < 0.01
