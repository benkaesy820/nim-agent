"""
Core module for NIM CLI.

This module contains the foundational components:
- Configuration management
- API client with resilience
- Streaming engine
- Error handling
- Metrics collection
"""

from nim_cli.core.config import (
    Config,
    ConfigManager,
    ModelConfig,
    NetworkConfig,
    DisplayConfig,
    BehaviorConfig,
    LoggingConfig,
    get_config,
    get_config_manager,
)
from nim_cli.core.client import (
    NIMClient,
    ChatResponse,
    create_client,
    setup_uvloop,
)
from nim_cli.core.errors import (
    NIMCLIError,
    ConfigError,
    ConfigFileError,
    ConfigValidationError,
    APIKeyError,
    NIMAPIError,
    NIMConnectionError,
    NIMTimeoutError,
    NIMRateLimitError,
    NIMAuthenticationError,
    NIMContentFilterError,
    NIMModelError,
    StreamError,
    StreamInterruptedError,
    StreamParseError,
    InputError,
    CommandError,
    is_retryable_error,
    get_error_hint,
)
from nim_cli.core.metrics import (
    MetricsCollector,
    RequestMetrics,
    RequestTracker,
    get_metrics,
)
from nim_cli.core.retry import (
    RetryPolicy,
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    RateLimitHandler,
    ResilientClient,
    create_default_retry_policy,
    create_default_circuit_breaker,
    create_default_rate_limiter,
)
from nim_cli.core.streaming import (
    StreamChunk,
    StreamProcessor,
    StreamMetrics,
)

__all__ = [
    # Config
    "Config",
    "ConfigManager",
    "ModelConfig",
    "NetworkConfig",
    "DisplayConfig",
    "BehaviorConfig",
    "LoggingConfig",
    "get_config",
    "get_config_manager",
    
    # Client
    "NIMClient",
    "ChatResponse",
    "create_client",
    "setup_uvloop",
    
    # Errors
    "NIMCLIError",
    "ConfigError",
    "ConfigFileError",
    "ConfigValidationError",
    "APIKeyError",
    "NIMAPIError",
    "NIMConnectionError",
    "NIMTimeoutError",
    "NIMRateLimitError",
    "NIMAuthenticationError",
    "NIMContentFilterError",
    "NIMModelError",
    "StreamError",
    "StreamInterruptedError",
    "StreamParseError",
    "InputError",
    "CommandError",
    "is_retryable_error",
    "get_error_hint",
    
    # Metrics
    "MetricsCollector",
    "RequestMetrics",
    "RequestTracker",
    "get_metrics",
    
    # Retry
    "RetryPolicy",
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "RateLimitHandler",
    "ResilientClient",
    "create_default_retry_policy",
    "create_default_circuit_breaker",
    "create_default_rate_limiter",
    
    # Streaming
    "StreamChunk",
    "StreamProcessor",
    "StreamMetrics",
]
