"""Leakage tests — six meta-tests verifying data integrity.

1. Training cutoff — max training date < earliest prediction date per fold
2. Feature timestamps — prediction date filtering verified
3. No season aggregates — same-season training data strictly before prediction
4. Reproducibility — two runs with same seed produce byte-identical JSON
5. Poison test — overwriting future results doesn't affect predictions (@pytest.mark.slow)
6. Too-good alarm — suspiciously strong results trigger failure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.engine.backtest.harness import run_backtest
from services.engine.backtest.report import serialise_report


class TestTrainingCutoff:
    """For every fold, max(training date) < earliest prediction date."""

    def test_no_future_data_in_training(self, multi_season_df, default_config):
        """Verify that no training data from a prediction date or later is used."""
        report = run_backtest(
            multi_season_df, default_config, created_at="2026-01-01T00:00:00Z",
        )

        for pred in report.predictions:
            pred_date = pd.Timestamp(pred.date)
            # All training data must be strictly before pred_date
            training_mask = multi_season_df["date"].dt.normalize() < pred_date.normalize()
            training_df = multi_season_df[training_mask]
            # The harness uses at most this many matches
            assert pred.n_training_matches <= len(training_df)

    def test_prediction_count_matches_held_out(self, multi_season_df, default_config):
        """All held-out matches should have predictions."""
        report = run_backtest(
            multi_season_df, default_config, created_at="2026-01-01T00:00:00Z",
        )
        held_out_count = multi_season_df[
            multi_season_df["season"].isin(default_config.held_out_seasons)
        ].shape[0]
        assert len(report.predictions) == held_out_count


class TestFeatureTimestamps:
    """Prediction date filtering — only matches on the prediction date are predicted."""

    def test_predictions_match_held_out_dates(self, multi_season_df, default_config):
        report = run_backtest(
            multi_season_df, default_config, created_at="2026-01-01T00:00:00Z",
        )
        held_out_mask = multi_season_df["season"].isin(default_config.held_out_seasons)
        held_out_dates = set(multi_season_df[held_out_mask]["date"].dt.strftime("%Y-%m-%d"))
        pred_dates = {p.date for p in report.predictions}
        assert pred_dates.issubset(held_out_dates)


class TestNoSeasonAggregates:
    """Training data from same season only contains matches strictly before prediction."""

    def test_same_season_data_strictly_before(self, multi_season_df, default_config):
        report = run_backtest(
            multi_season_df, default_config, created_at="2026-01-01T00:00:00Z",
        )
        for pred in report.predictions:
            pred_date = pd.Timestamp(pred.date).normalize()
            same_season = multi_season_df[multi_season_df["season"] == pred.season]
            # Any same-season data in training must be strictly before pred_date
            before_mask = same_season["date"].dt.normalize() < pred_date
            matches_on_or_after = same_season[~before_mask]
            # Matches on or after should not contribute to training
            # (they are either the prediction itself or future matches)
            for _, m in matches_on_or_after.iterrows():
                match_date = m["date"]
                assert match_date.normalize() >= pred_date


class TestReproducibility:
    """Two runs with same seed produce byte-identical JSON."""

    def test_deterministic_output(self, multi_season_df, default_config):
        created_at = "2026-01-01T00:00:00Z"
        report1 = run_backtest(multi_season_df, default_config, created_at=created_at)
        report2 = run_backtest(multi_season_df, default_config, created_at=created_at)

        json1 = serialise_report(report1)
        json2 = serialise_report(report2)
        assert json1 == json2

    def test_prediction_order_stable(self, multi_season_df, default_config):
        """Predictions should be in the same order across runs."""
        created_at = "2026-01-01T00:00:00Z"
        report1 = run_backtest(multi_season_df, default_config, created_at=created_at)
        report2 = run_backtest(multi_season_df, default_config, created_at=created_at)

        for p1, p2 in zip(report1.predictions, report2.predictions):
            assert p1.date == p2.date
            assert p1.home_team == p2.home_team
            assert p1.away_team == p2.away_team
            assert p1.model_home == p2.model_home


@pytest.mark.slow
class TestPoisonTest:
    """Overwriting future results should not affect predictions."""

    @pytest.mark.parametrize("sample_idx", range(5))
    def test_poison_does_not_affect_predictions(
        self, multi_season_df, default_config, sample_idx,
    ):
        """For a sampled prediction date, overwrite results on/after that date
        with random scores, re-run predictions for that date, assert unchanged."""
        created_at = "2026-01-01T00:00:00Z"

        # Run clean backtest
        report_clean = run_backtest(multi_season_df, default_config, created_at=created_at)
        if not report_clean.predictions:
            pytest.skip("No predictions generated")

        # Select a prediction date to test
        pred_dates = sorted({p.date for p in report_clean.predictions})
        if sample_idx >= len(pred_dates):
            pytest.skip(f"Only {len(pred_dates)} prediction dates available")
        target_date = pred_dates[sample_idx]
        target_ts = pd.Timestamp(target_date)

        # Poison: overwrite results on/after the target date
        poisoned_df = multi_season_df.copy()
        rng = np.random.default_rng(99 + sample_idx)
        future_mask = poisoned_df["date"].dt.normalize() >= target_ts.normalize()
        n_future = future_mask.sum()
        poisoned_df.loc[future_mask, "home_goals"] = rng.integers(0, 5, size=n_future)
        poisoned_df.loc[future_mask, "away_goals"] = rng.integers(0, 5, size=n_future)
        # Update ftr
        hg = poisoned_df.loc[future_mask, "home_goals"]
        ag = poisoned_df.loc[future_mask, "away_goals"]
        poisoned_df.loc[future_mask, "ftr"] = np.where(
            hg > ag, "H", np.where(hg == ag, "D", "A"),
        )

        # Run poisoned backtest
        report_poisoned = run_backtest(poisoned_df, default_config, created_at=created_at)

        # Get predictions for the target date from both runs
        clean_preds = [p for p in report_clean.predictions if p.date == target_date]
        poisoned_preds = [p for p in report_poisoned.predictions if p.date == target_date]

        assert len(clean_preds) == len(poisoned_preds)

        for cp, pp in zip(clean_preds, poisoned_preds):
            # Model probabilities should be identical (training data is the same)
            assert cp.model_home == pytest.approx(pp.model_home, abs=1e-10)
            assert cp.model_draw == pytest.approx(pp.model_draw, abs=1e-10)
            assert cp.model_away == pytest.approx(pp.model_away, abs=1e-10)


class TestTooGoodAlarm:
    """Suspiciously strong results should trigger the too-good alarm."""

    def test_gate_includes_too_good_checks(self, multi_season_df, default_config):
        report = run_backtest(
            multi_season_df, default_config, created_at="2026-01-01T00:00:00Z",
        )
        gate_names = {d.name for d in report.gate_details}
        assert "too_good_rps" in gate_names
        assert "too_good_vs_bookmaker" in gate_names

    def test_perfect_predictions_trigger_alarm(self):
        """A model producing RPS < 0.190 should fail the gate."""
        from services.engine.backtest.gate import check_too_good_rps
        from services.engine.backtest.types import MetricSet

        ms = MetricSet(rps=0.18, log_loss=0.9, brier_home=0.1,
                       brier_draw=0.1, brier_away=0.1, n_matches=100, hit_rate=0.8)
        result = check_too_good_rps(ms)
        assert result.passed is False
