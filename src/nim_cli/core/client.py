"""
NVIDIA NIM API client with optimized connection handling.

This module provides:
- AsyncOpenAI-based client for NVIDIA NIM
- Connection pooling and keep-alive
- Streaming support
- Comprehensive error handling
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Optional,
    Sequence,
    Union,
)

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletion,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from nim_cli.core.config import Config, get_config
from nim_cli.core.errors import (
    NIMAPIError,
    NIMAuthenticationError,
    NIMConnectionError,
    NIMContentFilterError,
    NIMModelError,
    NIMRateLimitError,
    NIMTimeoutError,
)
from nim_cli.core.metrics import MetricsCollector, RequestTracker, get_metrics
from nim_cli.core.retry import (
    CircuitBreaker,
    CircuitOpenError,
    RateLimitHandler,
    ResilientClient,
    RetryPolicy,
    create_default_circuit_breaker,
    create_default_rate_limiter,
    create_default_retry_policy,
)
from nim_cli.core.streaming import StreamChunk


# =============================================================================
# Response Types
# =============================================================================


@dataclass
class ChatResponse:
    """Container for a complete chat response."""
    
    content: str
    model: str
    usage: dict[str, int]
    finish_reason: str
    latency_ms: float
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.usage.get("total_tokens", 0)
    
    @property
    def prompt_tokens(self) -> int:
        """Prompt tokens used."""
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        """Completion tokens generated."""
        return self.usage.get("completion_tokens", 0)


# =============================================================================
# NIM Client
# =============================================================================


class NIMClient:
    """
    Optimized client for NVIDIA NIM API.
    
    This client provides:
    - Connection pooling with keep-alive
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Rate limit handling
    - Streaming support
    - Metrics collection
    
    Usage:
        async with NIMClient(config) as client:
            # Non-streaming
            response = await client.chat("Hello!")
            
            # Streaming
            async for chunk in client.chat_stream("Tell me a story"):
                print(chunk.content, end="", flush=True)
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limiter: Optional[RateLimitHandler] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        """
        Initialize NIM client.
        
        Args:
            config: Configuration (uses global if not provided)
            retry_policy: Custom retry policy
            circuit_breaker: Custom circuit breaker
            rate_limiter: Custom rate limiter
            metrics: Custom metrics collector
        """
        self._config = config or get_config()
        self._metrics = metrics or get_metrics()
        
        # Initialize resilience components
        self._retry_policy = retry_policy or create_default_retry_policy()
        self._circuit_breaker = circuit_breaker or create_default_circuit_breaker()
        self._rate_limiter = rate_limiter or create_default_rate_limiter()
        
        # Client will be initialized lazily
        self._client: Optional[AsyncOpenAI] = None
        self._closed = False
    
    @property
    def model(self) -> str:
        """Get configured model name."""
        return self._config.model.name
    
    @property
    def is_closed(self) -> bool:
        """Check if client is closed."""
        return self._closed
    
    async def __aenter__(self) -> "NIMClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self) -> AsyncOpenAI:
        """Ensure client is initialized and return it."""
        if self._client is None or self._closed:
            api_key = self._config.get_api_key()
            
            # Create optimized client
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._config.network.base_url,
                timeout=self._create_timeout(),
                max_retries=0,  # We handle retries ourselves
            )
            self._closed = False
        
        return self._client
    
    def _create_timeout(self) -> Any:
        """Create timeout configuration for httpx."""
        import httpx
        
        return httpx.Timeout(
            connect=self._config.network.connect_timeout,
            read=self._config.network.read_timeout,
            write=self._config.network.timeout,
            pool=self._config.network.timeout,
        )
    
    async def close(self) -> None:
        """Close the client and release resources."""
        if self._client is not None and not self._closed:
            await self._client.close()
            self._closed = True
    
    # =========================================================================
    # Chat API
    # =========================================================================
    
    async def chat(
        self,
        message: str,
        *,
        system: Optional[str] = None,
        history: Optional[Sequence[ChatCompletionMessageParam]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """
        Send a chat message and get a complete response.

        Args:
            message: User message
            system: Optional system prompt
            history: Conversation history
            model: Override configured model
            temperature: Override configured temperature
            max_tokens: Override configured max tokens

        Returns:
            ChatResponse with content and metadata
        """
        messages = self._build_messages(message, system, history)
        model = model or self._config.model.name
        temperature = temperature if temperature is not None else self._config.model.temperature
        max_tokens = max_tokens or self._config.model.max_tokens

        start_time = time.monotonic()

        # Use `with` so __exit__ is called exactly once, correctly
        with self._metrics.track_request(model) as request_tracker:
            client = await self._ensure_client()

            response = await self._execute_with_resilience(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            latency_ms = (time.monotonic() - start_time) * 1000

            request_tracker.record_tokens(
                response.usage.prompt_tokens if response.usage else 0,
                sent=True,
            )
            request_tracker.record_tokens(
                response.usage.completion_tokens if response.usage else 0,
            )

            return ChatResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason or "stop",
                latency_ms=latency_ms,
            )
    
    async def chat_stream(
        self,
        message: str,
        *,
        system: Optional[str] = None,
        history: Optional[Sequence[ChatCompletionMessageParam]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Send a chat message and stream the response.

        The openai SDK already parses SSE events into structured ChatCompletionChunk
        objects. We convert those directly to StreamChunk — no re-encoding needed.

        Yields:
            StreamChunk objects with content
        """
        messages = self._build_messages(message, system, history)
        model = model or self._config.model.name
        temperature = temperature if temperature is not None else self._config.model.temperature
        max_tokens = max_tokens or self._config.model.max_tokens

        start_time = time.monotonic()
        accumulated = ""
        first_token_recorded = False

        # Use `with` so __exit__ is called exactly once, correctly
        with self._metrics.track_request(model) as request_tracker:
            client = await self._ensure_client()

            stream = await self._execute_with_resilience(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                content = delta.content or ""

                if content:
                    if not first_token_recorded:
                        request_tracker.record_first_token()
                        first_token_recorded = True
                    request_tracker.record_tokens(1)
                    accumulated += content

                    yield StreamChunk(
                        content=content,
                        is_complete=False,
                        token_count=1,
                        latency_ms=(time.monotonic() - start_time) * 1000,
                        accumulated_content=accumulated,
                    )

                if finish_reason and finish_reason not in ("null", None):
                    if finish_reason == "content_filter":
                        from nim_cli.core.errors import NIMContentFilterError
                        raise NIMContentFilterError(
                            reason="Response was filtered by content policy"
                        )
                    yield StreamChunk(
                        content="",
                        is_complete=True,
                        finish_reason=finish_reason,
                        accumulated_content=accumulated,
                    )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _build_messages(
        self,
        message: str,
        system: Optional[str],
        history: Optional[Sequence[ChatCompletionMessageParam]],
    ) -> list[ChatCompletionMessageParam]:
        """Build the messages list for the API."""
        messages: list[ChatCompletionMessageParam] = []
        
        # Add system prompt
        if system:
            messages.append({"role": "system", "content": system})
        elif self._config.behavior.system_prompt:
            messages.append({
                "role": "system",
                "content": self._config.behavior.system_prompt,
            })
        
        # Add history
        if history:
            messages.extend(history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    async def _execute_with_resilience(self, func: Any) -> Any:
        """Execute a function with retry and circuit breaker."""
        import httpx
        
        # Check circuit breaker
        if self._circuit_breaker.is_open:
            raise CircuitOpenError(
                "Circuit breaker is open - too many recent failures"
            )
        
        # Check rate limiter
        wait_time = self._rate_limiter.get_wait_time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        # Track rate limit for proactive throttling
        self._rate_limiter.record_request()
        
        last_error = None
        
        for attempt in range(self._retry_policy.max_attempts):
            try:
                result = await func()
                self._circuit_breaker.record_success()
                return result
                
            except Exception as e:
                last_error = e
                
                # Handle specific errors
                if self._is_authentication_error(e):
                    self._circuit_breaker.record_failure()
                    raise NIMAuthenticationError() from e
                
                if self._is_rate_limit_error(e):
                    retry_after = self._extract_retry_after(e)
                    self._rate_limiter.record_rate_limit(retry_after)
                    
                    if attempt < self._retry_policy.max_attempts - 1:
                        wait = retry_after or self._retry_policy.get_wait_time(attempt)
                        await asyncio.sleep(wait)
                        continue
                    raise NIMRateLimitError(retry_after=retry_after) from e
                
                if self._is_timeout_error(e):
                    self._circuit_breaker.record_failure()
                    if attempt < self._retry_policy.max_attempts - 1:
                        await asyncio.sleep(self._retry_policy.get_wait_time(attempt))
                        continue
                    raise NIMTimeoutError() from e
                
                if self._is_connection_error(e):
                    self._circuit_breaker.record_failure()
                    if attempt < self._retry_policy.max_attempts - 1:
                        await asyncio.sleep(self._retry_policy.get_wait_time(attempt))
                        continue
                    raise NIMConnectionError() from e
                
                if self._is_content_filter_error(e):
                    raise NIMContentFilterError() from e
                
                if self._is_model_error(e):
                    raise NIMModelError() from e
                
                # Unknown error - don't retry
                raise self._wrap_error(e)
        
        # Exhausted retries
        if last_error:
            raise self._wrap_error(last_error)
        raise NIMAPIError("Unknown error occurred")
    
    def _is_authentication_error(self, e: Exception) -> bool:
        """Check if error is authentication related."""
        error_str = str(e).lower()
        return "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str
    
    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Check if error is rate limit related."""
        error_str = str(e).lower()
        return "429" in error_str or "rate limit" in error_str or "too many requests" in error_str
    
    def _is_timeout_error(self, e: Exception) -> bool:
        """Check if error is timeout related."""
        import httpx
        return isinstance(e, (httpx.TimeoutException, asyncio.TimeoutError))
    
    def _is_connection_error(self, e: Exception) -> bool:
        """Check if error is connection related."""
        import httpx
        return isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError))
    
    def _is_content_filter_error(self, e: Exception) -> bool:
        """Check if error is content filter related."""
        error_str = str(e).lower()
        return "content_filter" in error_str or "content filter" in error_str
    
    def _is_model_error(self, e: Exception) -> bool:
        """Check if error is model related."""
        error_str = str(e).lower()
        return "model not found" in error_str or "404" in error_str
    
    def _extract_retry_after(self, e: Exception) -> Optional[float]:
        """Extract retry-after value from error."""
        # Try to extract from headers if available
        if hasattr(e, "response") and hasattr(e.response, "headers"):
            retry_after = e.response.headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return None
    
    def _wrap_error(self, e: Exception) -> NIMAPIError:
        """Wrap an unknown error in a NIMAPIError."""
        return NIMAPIError(f"API error: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================


@asynccontextmanager
async def create_client(
    config: Optional[Config] = None,
) -> AsyncGenerator[NIMClient, None]:
    """
    Create a client with automatic cleanup.
    
    Usage:
        async with create_client() as client:
            response = await client.chat("Hello!")
    """
    client = NIMClient(config)
    try:
        yield client
    finally:
        await client.close()


# =============================================================================
# Module-level setup for uvloop (performance optimization)
# =============================================================================


def setup_uvloop() -> bool:
    """
    Set up uvloop for better performance (2-4x faster).
    
    Returns:
        True if uvloop was set up, False otherwise
    """
    if sys.platform == "win32":
        # uvloop doesn't support Windows
        return False
    
    try:
        import uvloop
        # New API for uvloop 0.18+
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        return True
    except ImportError:
        return False


# Auto-setup on import (can be disabled by setting env var)
if os.environ.get("NIM_CLI_NO_UVLOOP", "").lower() not in ("1", "true", "yes"):
    setup_uvloop()
