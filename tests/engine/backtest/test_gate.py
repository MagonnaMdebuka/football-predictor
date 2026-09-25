"""Tests for gate pass/fail checks and too-good alarm."""

from __future__ import annotations

from services.engine.backtest.gate import (
    TOO_GOOD_RPS_THRESHOLD,
    check_beats_baseline,
    check_too_good_rps,
    check_too_good_vs_bookmaker,
    run_gate_checks,
)
from services.engine.backtest.types import MetricSet, SeasonMetrics


def _ms(rps: float = 0.20, log_loss: float = 1.02, **kw) -> MetricSet:
    defaults = dict(
        brier_home=0.22, brier_draw=0.31, brier_away=0.24,
        n_matches=100, hit_rate=0.53,
    )
    defaults.update(kw)
    return MetricSet(rps=rps, log_loss=log_loss, **defaults)


def _season_metrics(
    model_rps=0.200, model_ll=1.02,
    br_rps=0.210, br_ll=1.05,
    ip_rps=0.205, ip_ll=1.03,
    bk_rps=0.195, bk_ll=1.00,
    bookmaker_available=True,
) -> SeasonMetrics:
    model = _ms(rps=model_rps, log_loss=model_ll)
    base_rate = _ms(rps=br_rps, log_loss=br_ll)
    indep = _ms(rps=ip_rps, log_loss=ip_ll)
    bookmaker = _ms(rps=bk_rps, log_loss=bk_ll) if bookmaker_available else None
    return SeasonMetrics(
        season="combined",
        model=model, uniform=_ms(), base_rate=base_rate,
        independent_poisson=indep, ablation=None, bookmaker=bookmaker,
        bookmaker_exclusion_count=0, early_season=None,
    )


class TestCheckBeatsBaseline:
    """Model vs baseline comparison."""

    def test_passes_when_model_lower(self):
        model = _ms(rps=0.200)
        baseline = _ms(rps=0.210)
        result = check_beats_baseline(model, baseline, "base_rate", "rps")
        assert result.passed is True

    def test_fails_when_model_higher(self):
        model = _ms(rps=0.220)
        baseline = _ms(rps=0.210)
        result = check_beats_baseline(model, baseline, "base_rate", "rps")
        assert result.passed is False

    def test_fails_when_equal(self):
        """Equal is not beating — model must be strictly better."""
        model = _ms(rps=0.210)
        baseline = _ms(rps=0.210)
        result = check_beats_baseline(model, baseline, "base_rate", "rps")
        assert result.passed is False

    def test_log_loss_comparison(self):
        model = _ms(log_loss=1.00)
        baseline = _ms(log_loss=1.05)
        result = check_beats_baseline(model, baseline, "independent_poisson", "log_loss")
        assert result.passed is True

    def test_message_contains_values(self):
        model = _ms(rps=0.200)
        baseline = _ms(rps=0.210)
        result = check_beats_baseline(model, baseline, "base_rate", "rps")
        assert "0.200" in result.message
        assert "0.210" in result.message
        assert "base_rate" in result.message


class TestTooGoodRPS:
    """Too-good alarm based on RPS threshold."""

    def test_normal_rps_passes(self):
        model = _ms(rps=0.200)
        result = check_too_good_rps(model)
        assert result.passed is True

    def test_suspiciously_low_rps_fails(self):
        model = _ms(rps=0.185)
        result = check_too_good_rps(model)
        assert result.passed is False

    def test_threshold_boundary(self):
        """Exactly at threshold should pass (not strictly less)."""
        model = _ms(rps=TOO_GOOD_RPS_THRESHOLD)
        result = check_too_good_rps(model)
        assert result.passed is True

    def test_message_mentions_leakage(self):
        model = _ms(rps=0.180)
        result = check_too_good_rps(model)
        assert "suspected leakage" in result.message


class TestTooGoodVsBookmaker:
    """Too-good alarm when model beats bookmaker on log loss."""

    def test_normal_passes(self):
        model = _ms(log_loss=1.02)
        bookmaker = _ms(log_loss=1.00)
        result = check_too_good_vs_bookmaker(model, bookmaker)
        assert result.passed is True

    def test_beats_bookmaker_fails(self):
        model = _ms(log_loss=0.98)
        bookmaker = _ms(log_loss=1.00)
        result = check_too_good_vs_bookmaker(model, bookmaker)
        assert result.passed is False

    def test_no_bookmaker_skips(self):
        model = _ms(log_loss=0.98)
        result = check_too_good_vs_bookmaker(model, None)
        assert result.passed is True
        assert "skipped" in result.message


class TestRunGateChecks:
    """Full gate check integration."""

    def test_all_pass(self):
        sm = _season_metrics()
        passed, details = run_gate_checks(sm)
        assert passed is True
        assert len(details) == 6
        assert all(d.passed for d in details)

    def test_fails_when_model_worse_than_base_rate(self):
        sm = _season_metrics(model_rps=0.215, br_rps=0.210)
        passed, details = run_gate_checks(sm)
        assert passed is False

    def test_fails_when_model_worse_than_independent_poisson(self):
        sm = _season_metrics(model_ll=1.04, ip_ll=1.03)
        passed, details = run_gate_checks(sm)
        assert passed is False

    def test_fails_on_too_good_rps(self):
        sm = _season_metrics(model_rps=0.185)
        passed, details = run_gate_checks(sm)
        assert passed is False

    def test_no_bookmaker_still_runs(self):
        sm = _season_metrics(bookmaker_available=False)
        passed, details = run_gate_checks(sm)
        # Should still pass if other checks are fine
        assert passed is True
        assert len(details) == 6
