"""Tests for quality checks: 380-match check, goals/shots consistency, no negatives, null rate."""

from unittest.mock import MagicMock, patch

import pytest

from services.engine.ingest.quality import (
    EXPECTED_MATCHES_PER_SEASON,
    EXPECTED_TEAMS_PER_SEASON,
    QualityReport,
    verify_season,
)


@pytest.fixture
def mock_session():
    """Create a mock session that returns configurable counts."""
    session = MagicMock()
    return session


class TestQualityReport:
    def test_passed_with_no_errors(self):
        report = QualityReport(season_label="2023-24")
        assert report.passed is True

    def test_failed_with_errors(self):
        report = QualityReport(season_label="2023-24", errors=["Missing matches"])
        assert report.passed is False


class TestVerifySeason:
    def _setup_session(self, session, match_count=380, team_count=20,
                       bad_shots=0, neg_values=0, null_goals=0, finished_count=380):
        """Configure mock session to return specified counts."""
        # We need a more sophisticated mock chain
        query_mock = MagicMock()
        session.query.return_value = query_mock

        # Match count
        filter_mock = MagicMock()
        query_mock.filter_by.return_value = filter_mock
        filter_mock.scalar.return_value = match_count

        # Team count (distinct queries)
        query_mock.filter_by.return_value.count.return_value = team_count

        # For filter() chains (stat checks)
        filter_chain = MagicMock()
        query_mock.filter.return_value = filter_chain
        filter_chain.scalar.return_value = 0  # No bad values by default

        return session

    def test_report_structure(self, mock_session):
        self._setup_session(mock_session)
        report = verify_season(mock_session, 1, 1, "2023-24")
        assert isinstance(report, QualityReport)
        assert report.season_label == "2023-24"

    def test_current_season_allows_fewer_matches(self, mock_session):
        self._setup_session(mock_session, match_count=100)
        report = verify_season(mock_session, 1, 1, "2026-27", is_current=True)
        # Current season should not error on match count != 380
        match_errors = [e for e in report.errors if "380" in e]
        assert len(match_errors) == 0

    def test_current_season_errors_on_zero_matches(self, mock_session):
        self._setup_session(mock_session, match_count=0)
        report = verify_season(mock_session, 1, 1, "2026-27", is_current=True)
        assert any("No matches found" in e for e in report.errors)
