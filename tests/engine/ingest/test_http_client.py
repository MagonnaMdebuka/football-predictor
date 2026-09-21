"""Tests for HTTP client: 429 backoff, 5xx retry, budget exhaustion, circuit breaker."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from services.engine.ingest.config import IngestConfig
from services.engine.ingest.http_client import (
    BudgetExhaustedError,
    CircuitOpenError,
    RateLimitedClient,
)


@pytest.fixture
def config():
    cfg = IngestConfig()
    cfg.redis_url = "redis://localhost:6379/0"
    cfg.fd_org_rate_limit = 100  # High limit to avoid waiting in tests
    cfg.fd_org_daily_budget = 500
    return cfg


@pytest.fixture
def client(config):
    c = RateLimitedClient(
        config=config,
        source_name="test",
        requests_per_minute=1000,  # No rate limiting in tests
        daily_budget=100,
        max_retries=2,
        circuit_breaker_threshold=3,
    )
    yield c
    c.close()


class TestRateLimiting:
    @patch.object(httpx.Client, "get")
    @patch.object(RateLimitedClient, "_check_budget")
    @patch.object(RateLimitedClient, "_increment_budget")
    def test_successful_request(self, mock_budget_inc, mock_budget_check, mock_get, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        response = client.get("https://example.com/test")
        assert response.status_code == 200
        mock_get.assert_called_once()


class TestBackoff429:
    @patch("time.sleep")
    @patch.object(httpx.Client, "get")
    @patch.object(RateLimitedClient, "_check_budget")
    @patch.object(RateLimitedClient, "_increment_budget")
    def test_retries_on_429(self, mock_budget_inc, mock_budget_check, mock_get, mock_sleep, client):
        rate_response = MagicMock()
        rate_response.status_code = 429
        rate_response.headers = {"Retry-After": "1"}

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()

        mock_get.side_effect = [rate_response, ok_response]

        response = client.get("https://example.com/test")
        assert response.status_code == 200
        assert mock_get.call_count == 2


class TestBackoff5xx:
    @patch("time.sleep")
    @patch.object(httpx.Client, "get")
    @patch.object(RateLimitedClient, "_check_budget")
    @patch.object(RateLimitedClient, "_increment_budget")
    def test_retries_on_500(self, mock_budget_inc, mock_budget_check, mock_get, mock_sleep, client):
        error_response = MagicMock()
        error_response.status_code = 500

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()

        mock_get.side_effect = [error_response, ok_response]

        response = client.get("https://example.com/test")
        assert response.status_code == 200
        assert mock_get.call_count == 2


class TestBudgetExhaustion:
    def test_raises_when_budget_exhausted(self, client):
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"100"  # At or above budget
        client._redis = mock_redis

        with pytest.raises(BudgetExhaustedError):
            client.get("https://example.com/test")


class TestCircuitBreaker:
    @patch("time.sleep")
    @patch.object(httpx.Client, "get")
    @patch.object(RateLimitedClient, "_check_budget")
    def test_opens_after_threshold_failures(self, mock_budget, mock_get, mock_sleep, client):
        error_response = MagicMock()
        error_response.status_code = 500

        mock_get.return_value = error_response

        with pytest.raises(CircuitOpenError):
            client.get("https://example.com/test")

        assert client._circuit_open is True

    def test_rejects_when_open(self, client):
        client._circuit_open = True

        with pytest.raises(CircuitOpenError):
            client.get("https://example.com/test")

    def test_reset_circuit_breaker(self, client):
        client._circuit_open = True
        client._consecutive_failures = 5
        client.reset_circuit_breaker()
        assert client._circuit_open is False
        assert client._consecutive_failures == 0
