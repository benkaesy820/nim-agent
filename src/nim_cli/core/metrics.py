"""
Performance metrics collection for NIM CLI.

This module provides comprehensive metrics tracking for:
- API request latency
- Token throughput
- Error rates
- Circuit breaker state
- Resource usage

All metrics are thread-safe and designed for minimal overhead.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from nim_cli.core.errors import NIMAPIError


# =============================================================================
# Data Classes for Metrics
# =============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single API request."""
    
    start_time: float
    end_time: Optional[float] = None
    first_token_time: Optional[float] = None
    tokens_sent: int = 0
    tokens_received: int = 0
    success: bool = False
    error: Optional[str] = None
    model: Optional[str] = None
    cached: bool = False
    
    @property
    def latency_ms(self) -> Optional[float]:
        """Total request latency in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000
    
    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Token throughput rate."""
        if self.end_time is None or self.tokens_received == 0:
            return None
        duration = self.end_time - self.start_time
        if duration <= 0:
            return None
        return self.tokens_received / duration


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_sent: int = 0
    total_tokens_received: int = 0
    total_latency_ms: float = 0.0
    total_ttft_ms: float = 0.0  # Time to first token
    
    # For rolling averages
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    ttfts: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    throughputs: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_latency_ms(self) -> float:
        """Average request latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    @property
    def avg_ttft_ms(self) -> float:
        """Average time to first token."""
        if not self.ttfts:
            return 0.0
        return sum(self.ttfts) / len(self.ttfts)
    
    @property
    def avg_throughput(self) -> float:
        """Average tokens per second."""
        if not self.throughputs:
            return 0.0
        return sum(self.throughputs) / len(self.throughputs)
    
    @property
    def success_rate(self) -> float:
        """Success rate as a fraction."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Error rate as a fraction."""
        return 1.0 - self.success_rate


@dataclass
class ErrorMetrics:
    """Tracking for different error types."""
    
    connection_errors: int = 0
    timeout_errors: int = 0
    rate_limit_errors: int = 0
    authentication_errors: int = 0
    content_filter_errors: int = 0
    other_errors: int = 0
    
    @property
    def total_errors(self) -> int:
        """Total number of errors."""
        return (
            self.connection_errors
            + self.timeout_errors
            + self.rate_limit_errors
            + self.authentication_errors
            + self.content_filter_errors
            + self.other_errors
        )


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker state tracking."""
    
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: Optional[float] = None
    total_trips: int = 0


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """
    Thread-safe metrics collection and reporting.
    
    This class provides a centralized location for collecting and
    aggregating performance metrics across the application.
    
    Usage:
        collector = MetricsCollector()
        
        with collector.track_request() as req:
            # Make API call
            req.record_first_token()
            req.record_tokens(100)
        
        print(collector.get_summary())
    """
    
    def __init__(self, window_size: int = 100) -> None:
        """
        Initialize metrics collector.
        
        Args:
            window_size: Number of requests to keep for rolling averages
        """
        self._lock = Lock()
        self._window_size = window_size
        
        # Aggregated metrics
        self._api_metrics = AggregatedMetrics(
            latencies=deque(maxlen=window_size),
            ttfts=deque(maxlen=window_size),
            throughputs=deque(maxlen=window_size),
        )
        self._error_metrics = ErrorMetrics()
        self._circuit_metrics = CircuitBreakerMetrics()
        
        # Session tracking
        self._session_start = time.monotonic()
        self._active_requests: dict[int, RequestMetrics] = {}
        self._request_counter = 0
    
    def track_request(self, model: Optional[str] = None) -> "RequestTracker":
        """
        Create a context manager for tracking a request.
        
        Args:
            model: Model name being used
            
        Returns:
            A context manager that tracks the request
        """
        return RequestTracker(self, model)
    
    def start_request(self, model: Optional[str] = None) -> int:
        """
        Start tracking a new request.
        
        Args:
            model: Model name being used
            
        Returns:
            Request ID for later updates
        """
        with self._lock:
            request_id = self._request_counter
            self._request_counter += 1
            self._active_requests[request_id] = RequestMetrics(
                start_time=time.monotonic(),
                model=model,
            )
        return request_id
    
    def record_first_token(self, request_id: int) -> None:
        """Record the time of first token received."""
        with self._lock:
            if request_id in self._active_requests:
                self._active_requests[request_id].first_token_time = time.monotonic()
    
    def record_tokens(self, request_id: int, count: int, sent: bool = False) -> None:
        """
        Record token count.
        
        Args:
            request_id: Request ID from start_request
            count: Number of tokens
            sent: If True, these are tokens sent (not received)
        """
        with self._lock:
            if request_id in self._active_requests:
                if sent:
                    self._active_requests[request_id].tokens_sent += count
                else:
                    self._active_requests[request_id].tokens_received += count
    
    def end_request(
        self,
        request_id: int,
        success: bool = True,
        error: Optional[Exception] = None,
    ) -> None:
        """
        End tracking for a request.
        
        Args:
            request_id: Request ID from start_request
            success: Whether the request succeeded
            error: Optional exception that occurred
        """
        with self._lock:
            if request_id not in self._active_requests:
                return
            
            req = self._active_requests.pop(request_id)
            req.end_time = time.monotonic()
            req.success = success
            if error:
                req.error = type(error).__name__
            
            # Update aggregated metrics
            self._api_metrics.total_requests += 1
            if success:
                self._api_metrics.successful_requests += 1
            else:
                self._api_metrics.failed_requests += 1
            
            self._api_metrics.total_tokens_sent += req.tokens_sent
            self._api_metrics.total_tokens_received += req.tokens_received
            
            # Update rolling averages
            if req.latency_ms is not None:
                self._api_metrics.latencies.append(req.latency_ms)
            if req.time_to_first_token_ms is not None:
                self._api_metrics.ttfts.append(req.time_to_first_token_ms)
            if req.tokens_per_second is not None:
                self._api_metrics.throughputs.append(req.tokens_per_second)
            
            # Update error metrics
            if error:
                self._record_error(error)
    
    def _record_error(self, error: Exception) -> None:
        """Record an error in the appropriate category."""
        from nim_cli.core.errors import (
            NIMAuthenticationError,
            NIMConnectionError,
            NIMContentFilterError,
            NIMRateLimitError,
            NIMTimeoutError,
        )
        
        if isinstance(error, NIMConnectionError):
            self._error_metrics.connection_errors += 1
        elif isinstance(error, NIMTimeoutError):
            self._error_metrics.timeout_errors += 1
        elif isinstance(error, NIMRateLimitError):
            self._error_metrics.rate_limit_errors += 1
        elif isinstance(error, NIMAuthenticationError):
            self._error_metrics.authentication_errors += 1
        elif isinstance(error, NIMContentFilterError):
            self._error_metrics.content_filter_errors += 1
        else:
            self._error_metrics.other_errors += 1
    
    def update_circuit_breaker(
        self,
        state: str,
        failure_count: int = 0,
        success_count: int = 0,
    ) -> None:
        """
        Update circuit breaker state.
        
        Args:
            state: Current state (closed, open, half_open)
            failure_count: Current failure count
            success_count: Current success count
        """
        with self._lock:
            old_state = self._circuit_metrics.state
            self._circuit_metrics.state = state
            self._circuit_metrics.failure_count = failure_count
            self._circuit_metrics.success_count = success_count
            
            if state == "open" and old_state != "open":
                self._circuit_metrics.total_trips += 1
                self._circuit_metrics.last_failure_time = time.monotonic()
            
            if state != old_state:
                self._circuit_metrics.last_state_change = time.monotonic()
    
    def get_summary(self) -> dict:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with all metric values
        """
        with self._lock:
            session_duration = time.monotonic() - self._session_start
            
            return {
                "session": {
                    "duration_seconds": round(session_duration, 2),
                    "active_requests": len(self._active_requests),
                },
                "api": {
                    "total_requests": self._api_metrics.total_requests,
                    "successful": self._api_metrics.successful_requests,
                    "failed": self._api_metrics.failed_requests,
                    "success_rate": round(self._api_metrics.success_rate, 4),
                    "total_tokens_sent": self._api_metrics.total_tokens_sent,
                    "total_tokens_received": self._api_metrics.total_tokens_received,
                    "avg_latency_ms": round(self._api_metrics.avg_latency_ms, 2),
                    "avg_ttft_ms": round(self._api_metrics.avg_ttft_ms, 2),
                    "avg_throughput_tps": round(self._api_metrics.avg_throughput, 2),
                },
                "errors": {
                    "total": self._error_metrics.total_errors,
                    "connection": self._error_metrics.connection_errors,
                    "timeout": self._error_metrics.timeout_errors,
                    "rate_limit": self._error_metrics.rate_limit_errors,
                    "authentication": self._error_metrics.authentication_errors,
                    "content_filter": self._error_metrics.content_filter_errors,
                    "other": self._error_metrics.other_errors,
                },
                "circuit_breaker": {
                    "state": self._circuit_metrics.state,
                    "failure_count": self._circuit_metrics.failure_count,
                    "total_trips": self._circuit_metrics.total_trips,
                },
            }
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._api_metrics = AggregatedMetrics(
                latencies=deque(maxlen=self._window_size),
                ttfts=deque(maxlen=self._window_size),
                throughputs=deque(maxlen=self._window_size),
            )
            self._error_metrics = ErrorMetrics()
            self._circuit_metrics = CircuitBreakerMetrics()
            self._session_start = time.monotonic()
            self._active_requests.clear()


# =============================================================================
# Request Tracker Context Manager
# =============================================================================


class RequestTracker:
    """
    Context manager for tracking a single request.
    
    Usage:
        with collector.track_request("llama-3.1") as req:
            response = client.chat(...)
            req.record_first_token()
            for token in response:
                req.record_tokens(1)
    """
    
    def __init__(
        self,
        collector: MetricsCollector,
        model: Optional[str] = None,
    ) -> None:
        self._collector = collector
        self._model = model
        self._request_id: Optional[int] = None
    
    def __enter__(self) -> "RequestTracker":
        self._request_id = self._collector.start_request(self._model)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._request_id is not None:
            self._collector.end_request(
                self._request_id,
                success=exc_type is None,
                error=exc_val,
            )
    
    def record_first_token(self) -> None:
        """Record that the first token has been received."""
        if self._request_id is not None:
            self._collector.record_first_token(self._request_id)
    
    def record_tokens(self, count: int, sent: bool = False) -> None:
        """Record token count."""
        if self._request_id is not None:
            self._collector.record_tokens(self._request_id, count, sent)


# =============================================================================
# Global instance
# =============================================================================

_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
