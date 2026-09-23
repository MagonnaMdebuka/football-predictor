"""Tests for report serialisation, deserialisation, and filename generation."""

from __future__ import annotations

import json

from services.engine.backtest.report import (
    FLOAT_PRECISION,
    SCHEMA_VERSION,
    deserialise_report,
    generate_filename,
    report_to_dict,
    reports_identical,
    serialise_report,
)
from services.engine.backtest.types import (
    BacktestConfig,
    BacktestReport,
    GateDetail,
    MatchPrediction,
    MetricSet,
    SeasonMetrics,
)


def _make_metric_set(**overrides) -> MetricSet:
    defaults = dict(
        rps=0.2012345678, log_loss=1.0234567890,
        brier_home=0.22, brier_draw=0.31, brier_away=0.24,
        n_matches=100, hit_rate=0.53,
    )
    defaults.update(overrides)
    return MetricSet(**defaults)


def _make_report() -> BacktestReport:
    config = BacktestConfig(
        held_out_seasons=("2024-25",),
        training_start_season="2019-20",
    )
    ms = _make_metric_set()
    season = SeasonMetrics(
        season="2024-25", model=ms, uniform=ms, base_rate=ms,
        independent_poisson=ms, bookmaker=None,
        bookmaker_exclusion_count=5, early_season=None,
    )
    pred = MatchPrediction(
        date="2024-08-17", season="2024-25",
        home_team="Alpha", away_team="Bravo",
        home_goals=2, away_goals=1, result="H", matchday=1,
        model_home=0.5, model_draw=0.25, model_away=0.25,
        uniform_home=1 / 3, uniform_draw=1 / 3, uniform_away=1 / 3,
        base_rate_home=0.45, base_rate_draw=0.27, base_rate_away=0.28,
        indep_poisson_home=0.48, indep_poisson_draw=0.26, indep_poisson_away=0.26,
        bookmaker_home=None, bookmaker_draw=None, bookmaker_away=None,
        n_training_matches=500,
    )
    gate = GateDetail(name="rps_vs_base_rate", passed=True, message="OK")
    return BacktestReport(
        schema_version=SCHEMA_VERSION,
        created_at="2026-09-22T12:00:00Z",
        git_commit="abc1234",
        config=config,
        seasons=[season],
        combined=season,
        early_season_combined=None,
        predictions=[pred],
        gate_passed=True,
        gate_details=[gate],
    )


class TestSerialisation:
    """JSON round-trip serialisation."""

    def test_round_trip(self):
        """Serialise and deserialise should produce equivalent report."""
        report = _make_report()
        json_str = serialise_report(report)
        restored = deserialise_report(json_str)

        assert restored.schema_version == report.schema_version
        assert restored.git_commit == report.git_commit
        assert restored.gate_passed == report.gate_passed
        assert len(restored.predictions) == len(report.predictions)
        assert restored.predictions[0].home_team == "Alpha"

    def test_unix_line_endings(self):
        report = _make_report()
        json_str = serialise_report(report)
        assert "\r\n" not in json_str
        assert json_str.endswith("\n")

    def test_sorted_keys(self):
        report = _make_report()
        json_str = serialise_report(report)
        d = json.loads(json_str)
        keys = list(d.keys())
        assert keys == sorted(keys)

    def test_float_precision(self):
        """Floats should be rounded to FLOAT_PRECISION decimal places."""
        report = _make_report()
        d = report_to_dict(report)
        # Check a metric value with many decimal places
        rps_str = str(d["combined"]["model"]["rps"])
        # Count decimal digits
        if "." in rps_str:
            decimal_part = rps_str.split(".")[1]
            assert len(decimal_part) <= FLOAT_PRECISION

    def test_numpy_types_converted(self):
        """numpy int64 and float64 should serialise without error."""
        report = _make_report()
        d = report_to_dict(report)
        # Should be valid JSON (no numpy types)
        json_str = json.dumps(d, sort_keys=True)
        assert isinstance(json_str, str)

    def test_deterministic_output(self):
        """Two serialisations of the same report should be identical."""
        report = _make_report()
        json1 = serialise_report(report)
        json2 = serialise_report(report)
        assert json1 == json2

    def test_config_round_trip(self):
        """Config fields should survive round-trip."""
        report = _make_report()
        json_str = serialise_report(report)
        restored = deserialise_report(json_str)
        assert restored.config.xi == report.config.xi
        assert restored.config.held_out_seasons == report.config.held_out_seasons
        assert restored.config.bookmaker_odds_cols.primary_home == "PSCH"


class TestFilenameGeneration:
    """Report filename format."""

    def test_format(self):
        filename = generate_filename(git_commit="abc1234")
        assert filename.endswith("_abc1234.json")
        # Should start with ISO timestamp
        parts = filename.split("_")
        assert parts[0].endswith("Z")

    def test_contains_timestamp(self):
        filename = generate_filename(git_commit="def5678")
        assert "T" in filename  # ISO format has T separator


class TestReportsIdentical:
    """Byte-identical comparison of two report files."""

    def test_identical_files(self, tmp_path):
        report = _make_report()
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        content = serialise_report(report)
        path_a.write_text(content, encoding="utf-8", newline="")
        path_b.write_text(content, encoding="utf-8", newline="")
        assert reports_identical(path_a, path_b)

    def test_different_files(self, tmp_path):
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text('{"a": 1}\n', encoding="utf-8")
        path_b.write_text('{"a": 2}\n', encoding="utf-8")
        assert not reports_identical(path_a, path_b)
