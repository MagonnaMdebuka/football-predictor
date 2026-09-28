"""Tests for predict runner utilities."""

from datetime import date, datetime, timezone

import numpy as np
import pytest

from services.engine.predict.utils import (
    compress_grid,
    compute_fingerprint,
    decompress_grid,
    flatten_markets,
)


class TestCompressGrid:
    def test_round_trip_preserves_values(self):
        """compress → decompress should return the original grid."""
        grid = np.random.default_rng(42).random((11, 11))
        grid /= grid.sum()
        compressed = compress_grid(grid)
        recovered = decompress_grid(compressed)
        np.testing.assert_allclose(recovered, grid, atol=1e-15)

    def test_compressed_is_bytes(self):
        """compress_grid returns bytes."""
        grid = np.random.default_rng(42).random((11, 11))
        grid /= grid.sum()
        compressed = compress_grid(grid)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    def test_compress_preserves_dtype(self):
        """Grid should be stored as float64 regardless of input dtype."""
        grid = np.ones((11, 11), dtype=np.float32) / 121
        recovered = decompress_grid(compress_grid(grid))
        assert recovered.dtype == np.float64


class TestComputeFingerprint:
    def test_deterministic(self):
        """Same inputs → same fingerprint."""
        fp1 = compute_fingerprint(1, 380, date(2026, 5, 11), "1.0.0", "abc123")
        fp2 = compute_fingerprint(1, 380, date(2026, 5, 11), "1.0.0", "abc123")
        assert fp1 == fp2

    def test_changes_with_different_inputs(self):
        """Different inputs → different fingerprints."""
        fp1 = compute_fingerprint(1, 380, date(2026, 5, 11), "1.0.0", "abc123")
        fp2 = compute_fingerprint(1, 381, date(2026, 5, 11), "1.0.0", "abc123")
        fp3 = compute_fingerprint(2, 380, date(2026, 5, 11), "1.0.0", "abc123")
        fp4 = compute_fingerprint(1, 380, date(2026, 5, 12), "1.0.0", "abc123")
        assert len({fp1, fp2, fp3, fp4}) == 4

    def test_changes_with_model_version(self):
        """Different model version → different fingerprint."""
        fp1 = compute_fingerprint(1, 380, date(2026, 5, 11), "1.0.0", "abc123")
        fp2 = compute_fingerprint(1, 380, date(2026, 5, 11), "1.1.0", "abc123")
        assert fp1 != fp2

    def test_changes_with_git_commit(self):
        """Different git commit → different fingerprint."""
        fp1 = compute_fingerprint(1, 380, date(2026, 5, 11), "1.0.0", "abc123")
        fp2 = compute_fingerprint(1, 380, date(2026, 5, 11), "1.0.0", "def456")
        assert fp1 != fp2

    def test_is_sha256_hex(self):
        """Fingerprint should be a 64-character hex string."""
        fp = compute_fingerprint(1, 100, date(2026, 1, 1), "1.0.0", "abc")
        assert len(fp) == 64
        int(fp, 16)  # raises if not valid hex

    def test_backward_compat_defaults(self):
        """Omitting model_version/git_commit gives a valid fingerprint."""
        fp = compute_fingerprint(1, 100, date(2026, 1, 1))
        assert len(fp) == 64


class TestFlattenMarkets:
    @pytest.fixture()
    def sample_markets(self):
        """Minimal markets dict matching grid_to_markets output structure."""
        return {
            "match_result": {"home": 0.45, "draw": 0.28, "away": 0.27},
            "double_chance": {
                "home_draw": 0.73, "draw_away": 0.55, "home_away": 0.72,
                "home": 0.45, "draw": 0.28, "away": 0.27,
            },
            "draw_no_bet": {"home": 0.625, "away": 0.375},
            "over_under": [
                {"line": 0.5, "over": 0.85, "under": 0.15},
                {"line": 2.5, "over": 0.52, "under": 0.48},
            ],
            "btts": {"yes": 0.48, "no": 0.52},
            "btts_over_25": {"yes": 0.35, "no": 0.65},
            "exact_total_goals": {"0": 0.05, "1": 0.12, "2": 0.22, "3": 0.25,
                                  "4": 0.18, "5": 0.10, "6+": 0.08},
            "correct_score": {"1-0": 0.10, "1-1": 0.09, "other": 0.81},
            "winning_margin": {"home_+1": 0.20, "draw": 0.28, "away_+1": 0.15,
                               "home_+2": 0.10, "away_+2": 0.07},
            "odd_even": {"odd": 0.49, "even": 0.51},
            "team_totals": {
                "home": [{"line": 0.5, "over": 0.70, "under": 0.30}],
                "away": [{"line": 0.5, "over": 0.60, "under": 0.40}],
            },
            "clean_sheet": {"home": 0.30, "away": 0.25},
            "win_to_nil": {"home": 0.18, "away": 0.12},
            "european_handicap": [
                {"line": 0, "home": 0.45, "draw": 0.28, "away": 0.27},
            ],
            "asian_handicap": [
                {"line": -0.5, "home": 0.45, "push": 0.0, "away": 0.55},
            ],
        }

    def test_returns_list_of_dicts(self, sample_markets):
        rows = flatten_markets(sample_markets, prediction_id=1)
        assert isinstance(rows, list)
        assert len(rows) > 0
        for row in rows:
            assert "market" in row
            assert "selection" in row
            assert "probability" in row

    def test_over_under_has_line(self, sample_markets):
        rows = flatten_markets(sample_markets, prediction_id=1)
        ou_rows = [r for r in rows if r["market"] == "over_under"]
        assert len(ou_rows) > 0
        for r in ou_rows:
            assert r["line"] is not None

    def test_match_result_has_no_line(self, sample_markets):
        rows = flatten_markets(sample_markets, prediction_id=1)
        mr_rows = [r for r in rows if r["market"] == "match_result"]
        assert len(mr_rows) == 3
        for r in mr_rows:
            assert r["line"] is None

    def test_prediction_id_propagated(self, sample_markets):
        rows = flatten_markets(sample_markets, prediction_id=42)
        for r in rows:
            assert r["prediction_id"] == 42
