"""Tests for odds extraction, fallback, and overround stripping."""

from __future__ import annotations

import json

import numpy as np
import pytest

from services.engine.backtest.odds import (
    batch_extract_probabilities,
    extract_odds,
    extract_probabilities,
    odds_to_probabilities,
    overround,
)
from services.engine.backtest.types import BookmakerOddsCols


class TestExtractOdds:
    """Extract decimal odds from raw JSON."""

    def test_primary_pinnacle_odds(self):
        raw = json.dumps({"PSCH": 2.10, "PSCD": 3.30, "PSCA": 3.50})
        result = extract_odds(raw)
        assert result == (2.10, 3.30, 3.50)

    def test_fallback_to_market_average(self):
        """When Pinnacle odds are missing, use market average."""
        raw = json.dumps({"AvgCH": 2.00, "AvgCD": 3.20, "AvgCA": 3.80})
        result = extract_odds(raw)
        assert result == (2.00, 3.20, 3.80)

    def test_primary_preferred_over_fallback(self):
        raw = json.dumps({
            "PSCH": 2.10, "PSCD": 3.30, "PSCA": 3.50,
            "AvgCH": 2.00, "AvgCD": 3.20, "AvgCA": 3.80,
        })
        result = extract_odds(raw)
        assert result == (2.10, 3.30, 3.50)

    def test_returns_none_for_missing_all_odds(self):
        raw = json.dumps({"HomeTeam": "Arsenal"})
        assert extract_odds(raw) is None

    def test_returns_none_for_invalid_json(self):
        assert extract_odds("not json") is None

    def test_returns_none_for_none_input(self):
        assert extract_odds(None) is None  # type: ignore[arg-type]

    def test_returns_none_for_odds_lte_one(self):
        """Odds <= 1.0 are invalid."""
        raw = json.dumps({"PSCH": 0.5, "PSCD": 3.30, "PSCA": 3.50})
        assert extract_odds(raw) is None

    def test_custom_column_names(self):
        cols = BookmakerOddsCols(
            primary_home="B365H", primary_draw="B365D", primary_away="B365A",
            fallback_home="AvgH", fallback_draw="AvgD", fallback_away="AvgA",
        )
        raw = json.dumps({"B365H": 1.80, "B365D": 3.50, "B365A": 4.50})
        result = extract_odds(raw, cols)
        assert result == (1.80, 3.50, 4.50)


class TestOddsToProbabilities:
    """Convert odds to probabilities with overround stripping."""

    def test_fair_odds_sum_to_one(self):
        """Fair odds (no overround) should convert cleanly."""
        p_h, p_d, p_a = odds_to_probabilities(2.0, 4.0, 4.0)
        assert p_h + p_d + p_a == pytest.approx(1.0, abs=1e-12)
        assert p_h == pytest.approx(0.5)
        assert p_d == pytest.approx(0.25)
        assert p_a == pytest.approx(0.25)

    def test_overround_stripped(self):
        """Odds with overround should still normalise to 1.0."""
        # These odds imply ~105% book
        p_h, p_d, p_a = odds_to_probabilities(1.90, 3.30, 4.20)
        assert p_h + p_d + p_a == pytest.approx(1.0, abs=1e-12)
        assert p_h > p_d > p_a  # home favourite preserved

    def test_proportional_normalisation(self):
        """Verify proportional method: p = (1/odds) / sum(1/odds)."""
        odds_h, odds_d, odds_a = 2.10, 3.30, 3.50
        implied = [1 / 2.10, 1 / 3.30, 1 / 3.50]
        total = sum(implied)
        expected = [x / total for x in implied]
        result = odds_to_probabilities(odds_h, odds_d, odds_a)
        for r, e in zip(result, expected):
            assert r == pytest.approx(e, abs=1e-12)


class TestOverround:
    """Calculate bookmaker overround."""

    def test_fair_book(self):
        assert overround(2.0, 4.0, 4.0) == pytest.approx(0.0, abs=1e-12)

    def test_typical_overround(self):
        vig = overround(1.90, 3.30, 4.20)
        assert vig > 0  # Bookmakers always have positive vig
        assert vig < 0.15  # Reasonable range


class TestExtractProbabilities:
    """End-to-end: JSON → probabilities."""

    def test_returns_normalised_probs(self):
        raw = json.dumps({"PSCH": 2.10, "PSCD": 3.30, "PSCA": 3.50})
        result = extract_probabilities(raw)
        assert result is not None
        assert sum(result) == pytest.approx(1.0, abs=1e-12)

    def test_returns_none_when_no_odds(self):
        assert extract_probabilities(json.dumps({})) is None


class TestBatchExtractProbabilities:
    """Batch extraction with exclusion counting."""

    def test_all_valid(self):
        jsons = [
            json.dumps({"PSCH": 2.10, "PSCD": 3.30, "PSCA": 3.50}),
            json.dumps({"PSCH": 1.80, "PSCD": 3.50, "PSCA": 4.50}),
        ]
        p_h, p_d, p_a, excluded = batch_extract_probabilities(jsons)
        assert excluded == 0
        assert p_h is not None
        assert len(p_h) == 2
        assert all(np.isfinite(p_h))

    def test_some_missing(self):
        jsons = [
            json.dumps({"PSCH": 2.10, "PSCD": 3.30, "PSCA": 3.50}),
            json.dumps({}),
            None,
        ]
        p_h, p_d, p_a, excluded = batch_extract_probabilities(jsons)
        assert excluded == 2
        assert p_h is not None
        assert np.isfinite(p_h[0])
        assert np.isnan(p_h[1])
        assert np.isnan(p_h[2])

    def test_all_missing(self):
        jsons = [json.dumps({}), None]
        p_h, p_d, p_a, excluded = batch_extract_probabilities(jsons)
        assert p_h is None
        assert excluded == 2
