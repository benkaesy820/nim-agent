"""
Retry logic and circuit breaker implementation for NIM CLI.

This module provides:
- Exponential backoff with jitter
- Circuit breaker pattern for fault tolerance
- Configurable retry policies
- Rate limit awareness
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from nim_cli.core.errors import (
    CircuitOpenError,
    NIMAPIError,
    NIMAuthenticationError,
    NIMContentFilterError,
    NIMRateLimitError,
    NIMTimeoutError,
    is_retryable_error,
)
from nim_cli.core.metrics import get_metrics

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation for API fault tolerance.
    
    The circuit breaker prevents cascading failures by:
    1. Tracking consecutive failures
    2. Opening after threshold is reached (fail fast)
    3. Allowing test requests after cooldown (half-open)
    4. Closing again if tests succeed
    
    Attributes:
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before trying half-open
        half_open_max_calls: Max test calls in half-open state
    """
    
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (should fail fast)."""
        if self._state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if self._should_attempt_recovery():
                self._transition_to_half_open()
                return False
            return True
        return False
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        return (time.monotonic() - self._last_failure_time) >= self.recovery_timeout
    
    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._update_metrics()
    
    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls += 1
            
            # Close circuit if enough successes in half-open
            if self._success_count >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0  # Reset on success
            
        self._update_metrics()
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self._last_failure_time = time.monotonic()
        
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            # Immediately open on failure in half-open
            self._state = CircuitState.OPEN
            
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                
        self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update metrics collector."""
        get_metrics().update_circuit_breaker(
            state=self._state.value,
            failure_count=self._failure_count,
            success_count=self._success_count,
        )
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._update_metrics()


# =============================================================================
# Retry Policy
# =============================================================================


@dataclass
class RetryPolicy:
    """
    Configuration for retry behavior.
    
    Uses exponential backoff with jitter to avoid thundering herd
    problems and provide fair retry distribution.
    """
    
    max_attempts: int = 3
    max_delay: float = 60.0
    base_delay: float = 1.0
    jitter_max: float = 1.0
    max_total_time: Optional[float] = None  # None = no limit
    
    def get_wait_time(self, attempt: int) -> float:
        """
        Calculate wait time for a given attempt.
        
        Uses exponential backoff: delay = base * 2^attempt + random_jitter
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Wait time in seconds
        """
        # Exponential backoff
        delay = self.base_delay * (2 ** attempt)
        
        # Add jitter (full jitter strategy)
        jitter = random.uniform(0, self.jitter_max)
        delay += jitter
        
        # Cap at max delay
        return min(delay, self.max_delay)
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if the operation should be retried
        """
        return is_retryable_error(exception)
    
    def create_retryer(self) -> AsyncRetrying:
        """Create a tenacity retryer with this policy."""
        # Build stop conditions
        stops = [stop_after_attempt(self.max_attempts)]
        if self.max_total_time is not None:
            stops.append(stop_after_delay(self.max_total_time))
        
        return AsyncRetrying(
            stop=stops[0] if len(stops) == 1 else stops[0] | stops[1],
            wait=wait_exponential_jitter(
                initial=self.base_delay,
                max=self.max_delay,
                jitter=self.jitter_max,
            ),
            retry=retry_if_exception(self.should_retry),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )


# =============================================================================
# Rate Limit Handler
# =============================================================================


@dataclass
class RateLimitHandler:
    """
    Handles API rate limiting with adaptive throttling.
    
    NVIDIA NIM free tier has approximately 40 requests per minute.
    This handler tracks usage and implements adaptive delays.
    """
    
    requests_per_minute: int = 40
    burst_allowance: int = 5  # Allow short bursts above limit
    
    _request_times: list[float] = field(default_factory=list, init=False)
    _last_rate_limit: Optional[float] = field(default=None, init=False)
    
    def record_request(self) -> None:
        """Record that a request was made."""
        now = time.monotonic()
        self._request_times.append(now)
        # Clean up old entries
        self._cleanup()
    
    def record_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """Record that we hit a rate limit."""
        self._last_rate_limit = time.monotonic()
        if retry_after:
            logger.warning(f"Rate limited. Retry after {retry_after}s")
    
    def _cleanup(self) -> None:
        """Remove entries older than 1 minute."""
        cutoff = time.monotonic() - 60.0
        self._request_times = [t for t in self._request_times if t > cutoff]
    
    def get_current_usage(self) -> int:
        """Get number of requests in the last minute."""
        self._cleanup()
        return len(self._request_times)
    
    def get_wait_time(self) -> float:
        """
        Get recommended wait time before next request.
        
        Returns:
            Recommended seconds to wait, or 0 if no wait needed
        """
        self._cleanup()
        current_usage = len(self._request_times)
        
        # If we're at or above limit, wait until oldest request ages out
        if current_usage >= self.requests_per_minute:
            if self._request_times:
                oldest = min(self._request_times)
                wait_time = 60.0 - (time.monotonic() - oldest) + 0.5
                return max(0, wait_time)
        
        # Proactive throttling as we approach limit
        if current_usage >= self.requests_per_minute - self.burst_allowance:
            # Add small delay to spread requests
            return 0.5
        
        return 0.0
    
    def should_wait(self) -> bool:
        """Check if we should wait before making a request."""
        return self.get_wait_time() > 0


# =============================================================================
# Resilient Client Wrapper
# =============================================================================


class ResilientClient:
    """
    Wraps an API client with retry and circuit breaker logic.
    
    This class provides a unified interface for resilient API calls
    that handle transient failures gracefully.
    """
    
    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ) -> None:
        """
        Initialize resilient client wrapper.
        
        Args:
            retry_policy: Retry configuration
            circuit_breaker: Circuit breaker instance
            rate_limit_handler: Rate limit handler
        """
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.rate_limiter = rate_limit_handler or RateLimitHandler()
    
    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with resilience patterns.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitOpenError: If circuit breaker is open
            RetryError: If all retries exhausted
            Original exception: From func if not retryable
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open:
            raise CircuitOpenError(
                f"Circuit breaker is open. Wait {self.circuit_breaker.recovery_timeout}s"
            )
        
        # Check rate limits
        wait_time = self.rate_limiter.get_wait_time()
        if wait_time > 0:
            logger.debug(f"Rate limit throttling: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        # Execute with retry
        retryer = self.retry_policy.create_retryer()
        
        try:
            result = await retryer(func, *args, **kwargs)
            self.circuit_breaker.record_success()
            self.rate_limiter.record_request()
            return result
            
        except NIMRateLimitError as e:
            self.rate_limiter.record_rate_limit(
                e.details.get("retry_after_seconds")
            )
            raise
            
        except Exception as e:
            if is_retryable_error(e):
                self.circuit_breaker.record_failure()
            raise


# =============================================================================
# Convenience Functions
# =============================================================================


def create_default_retry_policy() -> RetryPolicy:
    """Create a retry policy with sensible defaults for NVIDIA NIM."""
    return RetryPolicy(
        max_attempts=3,
        max_delay=60.0,
        base_delay=1.0,
        jitter_max=1.0,
        max_total_time=180.0,  # 3 minutes max
    )


def create_default_circuit_breaker() -> CircuitBreaker:
    """Create a circuit breaker with sensible defaults."""
    return CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_max_calls=3,
    )


def create_default_rate_limiter() -> RateLimitHandler:
    """Create a rate limiter for NVIDIA NIM free tier."""
    return RateLimitHandler(
        requests_per_minute=40,
        burst_allowance=5,
    )
