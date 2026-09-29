"""Tests for the accuracy API endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_accuracy_metric(
    metric_name="brier", metric_value=0.22, sample_size=760,
    market="match_result_home", source="backtest",
):
    m = MagicMock()
    m.metric_name = metric_name
    m.metric_value = metric_value
    m.sample_size = sample_size
    m.market = market
    m.source = source
    return m


def _make_calibration_map(
    market="match_result", selection="home",
    bin_lower=0.0, bin_upper=0.1,
    predicted_frequency=0.05, observed_frequency=0.04,
    sample_size=76, is_current=True,
):
    c = MagicMock()
    c.market = market
    c.selection = selection
    c.bin_lower = bin_lower
    c.bin_upper = bin_upper
    c.predicted_frequency = predicted_frequency
    c.observed_frequency = observed_frequency
    c.sample_size = sample_size
    c.is_current = is_current
    return c


def _make_model_vs_market(
    market="match_result", selection="home",
    model_prob=0.55, bookmaker_prob=0.50,
    source="backtest",
):
    r = MagicMock()
    r.market = market
    r.selection = selection
    r.model_prob = model_prob
    r.bookmaker_prob = bookmaker_prob
    r.bookmaker_source = "average"
    r.source = source
    return r


@pytest.mark.asyncio
async def test_accuracy_overview_returns_200(client, mock_db_session):
    """GET /api/v1/accuracy returns 200 with market list."""
    metrics = [
        _make_accuracy_metric("brier", 0.22, 760, "match_result_home"),
        _make_accuracy_metric("hit_rate", 0.48, 760, "match_result_home"),
    ]
    cal_maps = [
        _make_calibration_map(bin_lower=i / 10, bin_upper=(i + 1) / 10)
        for i in range(10)
    ]

    call_count = 0

    async def mock_execute(stmt):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            # AccuracyMetric query
            result.scalars.return_value.all.return_value = metrics
        else:
            # CalibrationMap query
            result.scalars.return_value.all.return_value = cal_maps
        return result

    mock_db_session.execute = mock_execute

    response = await client.get("/api/v1/accuracy")
    assert response.status_code == 200
    data = response.json()
    assert "markets" in data
    assert "total_settled" in data
    assert "source" in data


@pytest.mark.asyncio
async def test_accuracy_empty_db(client, mock_db_session):
    """GET /api/v1/accuracy with empty DB returns valid empty response."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/accuracy")
    assert response.status_code == 200
    data = response.json()
    assert data["markets"] == []
    assert data["total_settled"] == 0


@pytest.mark.asyncio
async def test_market_detail_returns_200(client, mock_db_session):
    """GET /api/v1/accuracy/{market} returns 200 with reliability bins."""
    cal_maps = [
        _make_calibration_map(bin_lower=i / 10, bin_upper=(i + 1) / 10)
        for i in range(10)
    ]
    metrics = [
        _make_accuracy_metric("brier", 0.22, 760, "match_result_home"),
    ]

    call_count = 0

    async def mock_execute(stmt):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            result.scalars.return_value.all.return_value = cal_maps
        else:
            result.scalars.return_value.all.return_value = metrics
        return result

    mock_db_session.execute = mock_execute

    response = await client.get("/api/v1/accuracy/match_result_home")
    assert response.status_code == 200
    data = response.json()
    assert "reliability" in data
    assert "calibration_bins" in data
    assert data["market"] == "match_result_home"


@pytest.mark.asyncio
async def test_unknown_market_returns_404(client, mock_db_session):
    """GET /api/v1/accuracy/{market} with unknown market returns 404."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/accuracy/nonexistent_market")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_model_vs_market_returns_200(client, mock_db_session):
    """GET /api/v1/accuracy/model-vs-market returns comparison data."""
    mvm_rows = [
        _make_model_vs_market("match_result", "home", 0.55, 0.50),
        _make_model_vs_market("match_result", "home", 0.40, 0.42),
    ]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mvm_rows
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/api/v1/accuracy/model-vs-market")
    assert response.status_code == 200
    data = response.json()
    assert "comparisons" in data
    assert "source" in data
