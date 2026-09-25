"""Tests for count model gate checks."""

from __future__ import annotations

from services.engine.backtest.count_gate import (
    TOO_GOOD_RATIO,
    _compute_baseline_brier,
    _compute_model_gate_brier,
    _gate_lines_for_model,
    check_beats_baseline_brier,
    check_too_good_brier,
    run_count_gate_checks,
)
from services.engine.backtest.count_types import CountMetricSet, CountPrediction
from services.engine.markets.counts import (
    BOOKING_GATE_LINES,
    BOOKING_MATCH_LINES,
    CORNER_GATE_LINES,
    CORNER_MATCH_LINES,
)


def _make_count_pred(
    actual_home: int = 5,
    actual_away: int = 4,
    mu_home: float = 5.0,
    mu_away: float = 4.0,
    alpha: float = 0.15,
    model_type: str = "corners",
    match_lines: list[float] | None = None,
) -> CountPrediction:
    """Helper to create a CountPrediction with O/U entries."""
    if match_lines is None:
        match_lines = CORNER_MATCH_LINES

    total = mu_home + mu_away
    match_ou = []
    for line in match_lines:
        # Simple approximation: P(over) roughly based on how far total is from line
        p_over = max(0.05, min(0.95, 0.5 + 0.1 * (total - line)))
        match_ou.append({"line": line, "over": p_over, "under": 1.0 - p_over})

    return CountPrediction(
        date="2024-08-17",
        season="2024-25",
        home_team="Alpha",
        away_team="Bravo",
        model_type=model_type,
        actual_home=actual_home,
        actual_away=actual_away,
        mu_home=mu_home,
        mu_away=mu_away,
        alpha=alpha,
        match_over_under=match_ou,
        home_over_under=[],
        away_over_under=[],
        n_training_matches=500,
    )


def _make_metrics(
    mean_brier: float = 0.230,
    n: int = 100,
    per_line_brier: dict[float, float] | None = None,
) -> CountMetricSet:
    if per_line_brier is None:
        per_line_brier = {
            line: mean_brier + (i - 3) * 0.005
            for i, line in enumerate(CORNER_MATCH_LINES)
        }
    return CountMetricSet(
        mean_brier=mean_brier,
        per_line_brier=per_line_brier,
        n_predictions=n,
        mean_predicted_total=9.0,
        mean_actual_total=9.2,
    )


class TestGateLineConstants:
    """Gate line constants are proper subsets of match lines."""

    def test_corner_gate_lines_subset(self):
        for line in CORNER_GATE_LINES:
            assert line in CORNER_MATCH_LINES

    def test_booking_gate_lines_subset(self):
        for line in BOOKING_GATE_LINES:
            assert line in BOOKING_MATCH_LINES

    def test_corner_gate_excludes_extremes(self):
        assert 7.5 not in CORNER_GATE_LINES
        assert 13.5 not in CORNER_GATE_LINES

    def test_booking_gate_excludes_extremes(self):
        assert 60.5 not in BOOKING_GATE_LINES

    def test_gate_lines_for_corners(self):
        assert _gate_lines_for_model("corners") == CORNER_GATE_LINES

    def test_gate_lines_for_cards(self):
        assert _gate_lines_for_model("cards") == BOOKING_GATE_LINES


class TestComputeBaselineBrier:
    """Constant-lambda baseline Brier computation."""

    def test_empty_predictions(self):
        assert _compute_baseline_brier([], [7.5, 8.5]) == 0.0

    def test_empty_lines(self):
        pred = _make_count_pred()
        assert _compute_baseline_brier([pred], []) == 0.0

    def test_positive_result(self):
        """Varied totals produce positive baseline Brier."""
        preds = [
            _make_count_pred(actual_home=a, actual_away=b)
            for a, b in [(3, 2), (5, 6), (4, 3), (7, 5), (2, 4),
                         (6, 3), (4, 7), (3, 3), (5, 4), (8, 3)]
        ]
        result = _compute_baseline_brier(preds, [7.5, 8.5, 9.5])
        assert result > 0

    def test_perfect_baseline_when_all_same(self):
        """If all totals are identical, baseline has zero Brier for any line."""
        preds = [_make_count_pred(actual_home=5, actual_away=5) for _ in range(20)]
        result = _compute_baseline_brier(preds, [9.5, 10.5])
        assert result == 0.0

    def test_varied_totals_give_nonzero(self):
        """Mixed totals produce non-zero baseline Brier."""
        preds = [
            _make_count_pred(actual_home=3, actual_away=2),  # total=5
            _make_count_pred(actual_home=7, actual_away=6),  # total=13
        ]
        result = _compute_baseline_brier(preds, [9.5])
        assert result > 0


class TestComputeModelGateBrier:
    """Model gate Brier uses gate lines only."""

    def test_uses_gate_lines_only(self):
        """Gate Brier should average only gate-line scores, not all lines."""
        # Create metrics where gate lines have different Brier from extremes
        per_line = {
            7.5: 0.50,   # extreme — excluded from gate
            8.5: 0.22,   # gate line
            9.5: 0.23,   # gate line
            10.5: 0.24,  # gate line
            11.5: 0.25,  # gate line
            12.5: 0.26,  # gate line
            13.5: 0.50,  # extreme — excluded from gate
        }
        metrics = _make_metrics(per_line_brier=per_line)
        result = _compute_model_gate_brier(metrics, CORNER_GATE_LINES)
        expected = (0.22 + 0.23 + 0.24 + 0.25 + 0.26) / 5
        assert abs(result - expected) < 1e-10

    def test_gate_excludes_7_5_and_13_5(self):
        """Gate Brier for corners excludes 7.5 and 13.5."""
        per_line = {line: 0.20 for line in CORNER_MATCH_LINES}
        per_line[7.5] = 0.90   # would ruin mean if included
        per_line[13.5] = 0.90  # would ruin mean if included
        metrics = _make_metrics(per_line_brier=per_line)
        result = _compute_model_gate_brier(metrics, CORNER_GATE_LINES)
        assert result < 0.25  # extremes excluded


class TestCheckBeatsBaselineBrier:
    """Model vs baseline Brier comparison."""

    def test_passes_when_model_lower(self):
        result = check_beats_baseline_brier(0.220, 0.240, "corners")
        assert result.passed is True
        assert "corners" in result.name

    def test_fails_when_model_higher(self):
        result = check_beats_baseline_brier(0.250, 0.240, "corners")
        assert result.passed is False

    def test_fails_when_equal(self):
        result = check_beats_baseline_brier(0.240, 0.240, "cards")
        assert result.passed is False

    def test_message_contains_values(self):
        result = check_beats_baseline_brier(0.220, 0.240, "corners")
        assert "0.220" in result.message
        assert "0.240" in result.message


class TestCheckTooGoodBrier:
    """Too-good alarm for count models (relative to baseline)."""

    def test_relative_too_good_passes_at_0_95(self):
        """Model at 0.95x baseline passes (above the 0.9 threshold)."""
        baseline = 0.240
        model = 0.95 * baseline  # ratio = 0.95
        result = check_too_good_brier(model, baseline, "corners")
        assert result.passed is True

    def test_relative_too_good_fails_at_0_85(self):
        """Model at 0.85x baseline fails (below the 0.9 threshold)."""
        baseline = 0.240
        model = 0.85 * baseline  # ratio = 0.85
        result = check_too_good_brier(model, baseline, "corners")
        assert result.passed is False
        assert "suspected leakage" in result.message

    def test_lopsided_market_no_false_alarm(self):
        """73/27 split market: both model and baseline have low Brier, passes."""
        # On a lopsided market the baseline Brier is low (e.g. 0.19)
        # and a good model might be at 0.185 — ratio 0.97, which passes
        baseline = 0.19
        model = 0.185  # ratio ~0.974
        result = check_too_good_brier(model, baseline, "cards")
        assert result.passed is True

    def test_zero_baseline_passes(self):
        """When baseline Brier is 0, too-good check is skipped."""
        result = check_too_good_brier(0.0, 0.0, "corners")
        assert result.passed is True

    def test_message_contains_ratio(self):
        baseline = 0.240
        model = 0.85 * baseline
        result = check_too_good_brier(model, baseline, "cards")
        assert "ratio=" in result.message


class TestRunCountGateChecks:
    """Full count gate integration."""

    def test_all_pass(self):
        preds = [
            _make_count_pred(actual_home=a, actual_away=b)
            for a, b in [(3, 2), (5, 6), (4, 3), (7, 5), (2, 4),
                         (6, 3), (4, 7), (3, 3), (5, 4), (8, 3)]
        ]
        # Compute what the baseline Brier actually is on gate lines
        baseline = _compute_baseline_brier(preds, CORNER_GATE_LINES)
        # Set model per-line Brier just below baseline for gate lines
        per_line = {line: baseline - 0.01 for line in CORNER_MATCH_LINES}
        metrics = _make_metrics(
            mean_brier=baseline - 0.01,
            per_line_brier=per_line,
        )
        passed, details = run_count_gate_checks(preds, metrics, "corners")
        assert passed is True
        assert len(details) == 2

    def test_empty_predictions_skips(self):
        metrics = _make_metrics(n=0)
        passed, details = run_count_gate_checks([], metrics, "corners")
        assert passed is True
        assert "skipped" in details[0].message

    def test_fails_when_worse_than_baseline(self):
        # All same totals => baseline Brier = 0, model can't beat it
        preds = [_make_count_pred(actual_home=5, actual_away=5) for _ in range(20)]
        metrics = _make_metrics(mean_brier=0.230)
        passed, details = run_count_gate_checks(preds, metrics, "corners")
        assert passed is False

    def test_model_type_in_names(self):
        preds = [_make_count_pred(actual_home=5, actual_away=4) for _ in range(5)]
        metrics = _make_metrics(mean_brier=0.220)
        _, details = run_count_gate_checks(preds, metrics, "cards")
        for d in details:
            assert "cards" in d.name

    def test_gate_uses_only_publishable_lines(self):
        """Gate Brier should exclude 7.5/13.5 for corners."""
        preds = [
            _make_count_pred(actual_home=a, actual_away=b)
            for a, b in [(3, 2), (5, 6), (4, 3), (7, 5), (2, 4),
                         (6, 3), (4, 7), (3, 3), (5, 4), (8, 3)]
        ]
        # Set extreme lines to terrible Brier but gate lines to good Brier
        baseline = _compute_baseline_brier(preds, CORNER_GATE_LINES)
        per_line = {}
        for line in CORNER_MATCH_LINES:
            if line in CORNER_GATE_LINES:
                per_line[line] = baseline - 0.01  # good
            else:
                per_line[line] = 0.99  # terrible
        metrics = _make_metrics(
            mean_brier=0.50,  # overall mean includes extremes
            per_line_brier=per_line,
        )
        passed, _ = run_count_gate_checks(preds, metrics, "corners")
        # Gate should pass because it only looks at gate lines
        assert passed is True
