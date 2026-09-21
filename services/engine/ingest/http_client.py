"""Rate-limited HTTP client with token bucket, backoff, circuit breaker, and Redis daily budget."""

import logging
import random
import time
from datetime import date

import httpx
import redis

from services.engine.ingest.config import IngestConfig

logger = logging.getLogger(__name__)


class BudgetExhaustedError(Exception):
    """Raised when the daily request budget for a source is exhausted."""


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open (too many recent failures)."""


class RateLimitedClient:
    """HTTP client with rate limiting, backoff, circuit breaker, and daily budget tracking.

    Token bucket enforces requests-per-minute. Exponential backoff with jitter handles
    429 and 5xx responses. A Redis-backed daily budget counter prevents exceeding quotas.
    Circuit breaker opens after consecutive failures to avoid hammering a broken service.
    """

    def __init__(
        self,
        config: IngestConfig | None = None,
        source_name: str = "football-data.org",
        requests_per_minute: int | None = None,
        daily_budget: int | None = None,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
    ) -> None:
        self.config = config or IngestConfig()
        self.source_name = source_name
        self.requests_per_minute = requests_per_minute or self.config.fd_org_rate_limit
        self.daily_budget = daily_budget or self.config.fd_org_daily_budget
        self.max_retries = max_retries
        self.circuit_breaker_threshold = circuit_breaker_threshold

        # Token bucket state
        self._min_interval = 60.0 / self.requests_per_minute
        self._last_request_time: float = 0.0

        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False

        # Redis for daily budget
        self._redis: redis.Redis | None = None

        # HTTP client
        self._client = httpx.Client(timeout=30.0)

    def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self.config.redis_url)
        return self._redis

    def _budget_key(self) -> str:
        return f"ingest:budget:{self.source_name}:{date.today().isoformat()}"

    def _check_budget(self) -> None:
        """Check Redis daily budget counter. Raise if exhausted."""
        try:
            r = self._get_redis()
            key = self._budget_key()
            count = r.get(key)
            if count is not None and int(count) >= self.daily_budget:
                raise BudgetExhaustedError(
                    f"Daily budget of {self.daily_budget} requests exhausted for {self.source_name}"
                )
        except redis.ConnectionError:
            logger.warning("Redis unavailable for budget check — proceeding without budget tracking")

    def _increment_budget(self) -> None:
        """Increment the Redis daily budget counter."""
        try:
            r = self._get_redis()
            key = self._budget_key()
            pipe = r.pipeline()
            pipe.incr(key)
            pipe.expire(key, 86400)  # Auto-expire after 24h
            pipe.execute()
        except redis.ConnectionError:
            logger.warning("Redis unavailable — skipping budget increment")

    def _wait_for_rate_limit(self) -> None:
        """Wait until the token bucket allows the next request."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def get(self, url: str, headers: dict | None = None) -> httpx.Response:
        """Make a rate-limited GET request with retries, backoff, and circuit breaker.

        Raises CircuitOpenError if the circuit breaker is open.
        Raises BudgetExhaustedError if the daily budget is exhausted.
        """
        if self._circuit_open:
            raise CircuitOpenError(
                f"Circuit breaker open for {self.source_name} after "
                f"{self.circuit_breaker_threshold} consecutive failures"
            )

        self._check_budget()

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()

            try:
                response = self._client.get(url, headers=headers or {})

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    wait = min(retry_after, 120) + random.uniform(0, 2)
                    logger.warning(
                        "Rate limited (429) on %s, waiting %.1fs (attempt %d/%d)",
                        url, wait, attempt + 1, self.max_retries + 1,
                    )
                    time.sleep(wait)
                    continue

                if response.status_code >= 500:
                    backoff = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Server error %d on %s, backing off %.1fs (attempt %d/%d)",
                        response.status_code, url, backoff, attempt + 1, self.max_retries + 1,
                    )
                    time.sleep(backoff)
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self.circuit_breaker_threshold:
                        self._circuit_open = True
                        raise CircuitOpenError(
                            f"Circuit breaker opened after {self._consecutive_failures} failures"
                        )
                    continue

                response.raise_for_status()
                self._consecutive_failures = 0
                self._increment_budget()
                return response

            except httpx.HTTPStatusError:
                raise
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_error = exc
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.circuit_breaker_threshold:
                    self._circuit_open = True
                    raise CircuitOpenError(
                        f"Circuit breaker opened after {self._consecutive_failures} failures"
                    ) from exc
                backoff = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Connection error on %s: %s, backing off %.1fs (attempt %d/%d)",
                    url, exc, backoff, attempt + 1, self.max_retries + 1,
                )
                time.sleep(backoff)

        raise last_error or httpx.ConnectError(f"Failed to connect to {url}")

    def close(self) -> None:
        """Close the HTTP client and Redis connection."""
        self._client.close()
        if self._redis is not None:
            self._redis.close()

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self._circuit_open = False
        self._consecutive_failures = 0
