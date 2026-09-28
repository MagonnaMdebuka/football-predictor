"""Tests for half-time gate checks."""

from __future__ import annotations

import pytest

from services.engine.backtest.halftime_gate import (
    _compute_ht_baselines,
    _compute_ht_metrics,
    run_halftime_gate_checks,
)
from services.engine.backtest.types import HalfTimePrediction


def _make_prediction(
    ht_hg=0, ht_ag=0, sh_hg=1, sh_ag=0,
    ht_model_home=0.4, ht_model_draw=0.35, ht_model_away=0.25,
    sh_model_home=0.45, sh_model_draw=0.30, sh_model_away=0.25,
    htft_probs=None,
) -> HalfTimePrediction:
    if htft_probs is None:
        htft_probs = {
            "HH": 0.20, "HD": 0.05, "HA": 0.03,
            "DH": 0.15, "DD": 0.12, "DA": 0.08,
            "AH": 0.10, "AD": 0.07, "AA": 0.20,
        }

    def _result(h, a):
        if h > a:
            return "H"
        if h == a:
            return "D"
        return "A"

    ft_hg = ht_hg + sh_hg
    ft_ag = ht_ag + sh_ag

    return HalfTimePrediction(
        date="2024-09-01",
        home_team="A",
        away_team="B",
        ht_home_goals=ht_hg,
        ht_away_goals=ht_ag,
        sh_home_goals=sh_hg,
        sh_away_goals=sh_ag,
        ht_result=_result(ht_hg, ht_ag),
        ft_result=_result(ft_hg, ft_ag),
        ht_model_home=ht_model_home,
        ht_model_draw=ht_model_draw,
        ht_model_away=ht_model_away,
        sh_model_home=sh_model_home,
        sh_model_draw=sh_model_draw,
        sh_model_away=sh_model_away,
        ht_lambda_home=0.5,
        ht_lambda_away=0.4,
        sh_lambda_home=0.7,
        sh_lambda_away=0.5,
        htft_probs=htft_probs,
    )


class TestComputeHtMetrics:
    def test_returns_valid_metrics(self):
        preds = [
            _make_prediction(ht_hg=1, ht_ag=0, sh_hg=1, sh_ag=1),
            _make_prediction(ht_hg=0, ht_ag=0, sh_hg=0, sh_ag=1),
            _make_prediction(ht_hg=0, ht_ag=1, sh_hg=2, sh_ag=0),
        ]
        metrics = _compute_ht_metrics(preds)
        assert metrics.n_predictions == 3
        assert 0.0 <= metrics.ht_rps <= 1.0
        assert 0.0 <= metrics.sh_rps <= 1.0
        assert metrics.ht_log_loss > 0


class TestComputeHtBaselines:
    def test_baseline_sums_to_one(self):
        preds = [
            _make_prediction(ht_hg=1, ht_ag=0),
            _make_prediction(ht_hg=0, ht_ag=0),
            _make_prediction(ht_hg=0, ht_ag=1),
        ]
        ht_br, sh_br = _compute_ht_baselines(preds)
        assert sum(ht_br) == pytest.approx(1.0, abs=1e-10)
        assert sum(sh_br) == pytest.approx(1.0, abs=1e-10)


class TestRunHalftimeGateChecks:
    def test_with_reasonable_predictions(self):
        """Gate should produce metrics and gate details."""
        preds = [_make_prediction() for _ in range(20)]
        metrics, passed, details = run_halftime_gate_checks(preds)
        assert metrics is not None
        assert metrics.n_predictions == 20
        assert isinstance(passed, bool)
        assert len(details) > 0

    def test_empty_predictions(self):
        metrics, passed, details = run_halftime_gate_checks([])
        assert metrics is None
        assert passed is True
        assert len(details) == 1
