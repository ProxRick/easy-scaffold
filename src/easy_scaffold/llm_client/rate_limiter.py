# src/easy_scaffold/llm_client/rate_limiter.py
import asyncio
import logging
import time
from typing import List, Optional

from ..configs.pydantic_models import RateLimitConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Per-minute rate limiter tracking:
    - Requests per minute (sliding window)
    - Input tokens per minute (token bucket)
    """

    def __init__(self, model: str, config: RateLimitConfig):
        self.model = model
        self.config = config
        self._request_times: List[float] = []  # Sliding window for requests
        self._token_bucket: float = config.input_tokens_per_minute  # Current tokens available
        self._last_refill: float = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """
        Wait until rate limit allows the request.
        Only checks request count here. Tokens are tracked after API call.
        """
        async with self._lock:
            await self._refill_bucket()
            await self._wait_for_request_slot()

    async def record_input_tokens(self, prompt_tokens: int):
        """
        Record actual input token usage after API call.
        Deducts from token bucket.
        """
        async with self._lock:
            self._token_bucket -= prompt_tokens
            # Ensure bucket doesn't go negative (shouldn't happen with proper limiting)
            if self._token_bucket < 0:
                self._token_bucket = 0

    async def check_token_availability(self, prompt_tokens: int) -> float:
        """
        Check if input tokens are available, return wait time if not.
        Returns 0 if available, wait time in seconds if not.
        """
        async with self._lock:
            await self._refill_bucket()
            if self._token_bucket >= prompt_tokens:
                return 0.0
            else:
                # Calculate wait time for refill
                needed = prompt_tokens - self._token_bucket
                wait_time = (needed / self.config.input_tokens_per_minute) * 60
                return wait_time + 0.1  # Small buffer

    async def _refill_bucket(self):
        """Refill token bucket based on time elapsed."""
        now = time.time()
        elapsed = now - self._last_refill

        if elapsed >= 60:
            # Full refill every 60 seconds
            self._token_bucket = self.config.input_tokens_per_minute
            self._last_refill = now
        elif elapsed > 0:
            # Partial refill based on elapsed time
            refill_amount = (elapsed / 60) * self.config.input_tokens_per_minute
            self._token_bucket = min(
                self.config.input_tokens_per_minute,
                self._token_bucket + refill_amount,
            )
            self._last_refill = now

    async def _wait_for_request_slot(self):
        """Wait if request rate limit is hit."""
        now = time.time()
        # Remove requests older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.config.requests_per_minute:
            # Wait until oldest request expires
            oldest = min(self._request_times)
            wait_time = 60 - (now - oldest) + 0.1  # Small buffer
            await asyncio.sleep(wait_time)
            # Retry after waiting
            await self._wait_for_request_slot()
        else:
            # Record this request
            self._request_times.append(now)


class RateLimiterRegistry:
    """Singleton registry for rate limiters per model."""

    _limiters: dict[str, RateLimiter] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get(
        cls, model: str, config: Optional[RateLimitConfig]
    ) -> Optional[RateLimiter]:
        """Get or create rate limiter for model."""
        if config is None:
            return None

        async with cls._lock:
            if model not in cls._limiters:
                cls._limiters[model] = RateLimiter(model, config)
            return cls._limiters[model]



