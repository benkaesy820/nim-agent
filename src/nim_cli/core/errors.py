"""
Custom exceptions for NIM CLI.

This module defines a comprehensive exception hierarchy for handling
all error conditions gracefully throughout the application.
"""

from typing import Any, Optional


class NIMCLIError(Exception):
    """
    Base exception for all NIM CLI errors.
    
    All custom exceptions inherit from this class to allow catching
    all application-specific errors with a single except clause.
    
    Attributes:
        message: Human-readable error description
        details: Additional context about the error
        recoverable: Whether the error can be recovered from
    """
    
    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
    
    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigError(NIMCLIError):
    """Base exception for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if config_path:
            details["config_path"] = config_path
        super().__init__(message, details=details, **kwargs)


class ConfigFileError(ConfigError):
    """Raised when configuration file cannot be read or parsed."""
    
    def __init__(
        self,
        message: str,
        *,
        config_path: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details["config_path"] = config_path
        if line is not None:
            details["line"] = line
        if column is not None:
            details["column"] = column
        super().__init__(message, details=details, **kwargs)


class ConfigValidationError(ConfigError):
    """Raised when configuration values fail validation."""
    
    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = repr(value)
        if expected:
            details["expected"] = expected
        super().__init__(message, details=details, **kwargs)


class APIKeyError(ConfigError):
    """Raised when API key is missing or invalid."""
    
    def __init__(
        self,
        message: str = "NVIDIA API key not found",
        *,
        env_var: str = "NVIDIA_API_KEY",
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details["env_var"] = env_var
        details["hint"] = "Set NVIDIA_API_KEY environment variable or configure in config file"
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# API/Network Errors
# =============================================================================


class NIMAPIError(NIMCLIError):
    """Base exception for NVIDIA NIM API errors."""
    
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if status_code is not None:
            details["status_code"] = status_code
        if request_id:
            details["request_id"] = request_id
        super().__init__(message, details=details, **kwargs)


class NIMConnectionError(NIMAPIError):
    """Raised when connection to NIM API fails."""
    
    def __init__(
        self,
        message: str = "Failed to connect to NVIDIA NIM API",
        *,
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, details=details, **kwargs)


class NIMTimeoutError(NIMAPIError):
    """Raised when API request times out."""
    
    def __init__(
        self,
        message: str = "Request to NVIDIA NIM API timed out",
        *,
        timeout: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if timeout is not None:
            details["timeout_seconds"] = timeout
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, **kwargs)


class NIMRateLimitError(NIMAPIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: Optional[float] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if retry_after is not None:
            details["retry_after_seconds"] = retry_after
        if limit is not None:
            details["limit"] = limit
        super().__init__(message, details=details, **kwargs)


class NIMAuthenticationError(NIMAPIError):
    """Raised when API authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed - check your API key",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, status_code=401, **kwargs)
        self.recoverable = False  # Authentication errors need user action


class NIMContentFilterError(NIMAPIError):
    """Raised when content is filtered by safety systems."""
    
    def __init__(
        self,
        message: str = "Content was filtered by safety systems",
        *,
        reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if reason:
            details["reason"] = reason
        super().__init__(message, details=details, **kwargs)


class NIMModelError(NIMAPIError):
    """Raised when requested model is unavailable."""
    
    def __init__(
        self,
        message: str = "Model not found or unavailable",
        *,
        model: Optional[str] = None,
        available_models: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model:
            details["requested_model"] = model
        if available_models:
            details["available_models"] = available_models
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Streaming Errors
# =============================================================================


class StreamError(NIMCLIError):
    """Base exception for streaming-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        partial_content: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if partial_content:
            details["partial_content_length"] = len(partial_content)
        self.partial_content = partial_content
        super().__init__(message, details=details, **kwargs)


class StreamInterruptedError(StreamError):
    """Raised when stream is interrupted unexpectedly."""
    
    def __init__(
        self,
        message: str = "Stream interrupted",
        *,
        bytes_received: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if bytes_received is not None:
            details["bytes_received"] = bytes_received
        super().__init__(message, **kwargs)


class StreamParseError(StreamError):
    """Raised when stream data cannot be parsed."""
    
    def __init__(
        self,
        message: str = "Failed to parse stream data",
        *,
        raw_data: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if raw_data:
            # Truncate to avoid huge error messages
            details["raw_data_preview"] = raw_data[:200] + "..." if len(raw_data) > 200 else raw_data
        super().__init__(message, **kwargs)


# =============================================================================
# Input/UI Errors
# =============================================================================


class InputError(NIMCLIError):
    """Raised when user input is invalid."""
    
    def __init__(
        self,
        message: str,
        *,
        input_value: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if input_value:
            details["input"] = input_value[:100]  # Truncate long input
        super().__init__(message, details=details, **kwargs)


class CommandError(NIMCLIError):
    """Raised when a command fails to execute."""
    
    def __init__(
        self,
        message: str,
        *,
        command: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if command:
            details["command"] = command
        super().__init__(message, details=details, **kwargs)


class CircuitOpenError(NIMCLIError):
    """Raised when circuit breaker is open."""
    
    def __init__(
        self,
        message: str = "Circuit breaker is open",
        *,
        retry_after: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if retry_after is not None:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Utility Functions
# =============================================================================


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the operation that caused the error can be retried
    """
    if isinstance(error, NIMRateLimitError):
        return True
    if isinstance(error, NIMConnectionError):
        return True
    if isinstance(error, NIMTimeoutError):
        return True
    if isinstance(error, StreamInterruptedError):
        return True
    if isinstance(error, NIMAuthenticationError):
        return False
    if isinstance(error, NIMContentFilterError):
        return False
    if isinstance(error, NIMCLIError):
        return error.recoverable
    return False


def get_error_hint(error: Exception) -> Optional[str]:
    """
    Get a helpful hint for resolving an error.
    
    Args:
        error: The exception to get a hint for
        
    Returns:
        A human-readable hint or None
    """
    hints = {
        APIKeyError: "Get your API key from https://build.nvidia.com",
        NIMRateLimitError: "Wait a moment and try again, or reduce request frequency",
        NIMTimeoutError: "Try using a smaller model or reducing max_tokens",
        NIMConnectionError: "Check your internet connection",
        NIMAuthenticationError: "Verify your API key is correct and active",
        NIMContentFilterError: "Rephrase your request to avoid flagged content",
        ConfigFileError: "Check your config file syntax",
    }
    
    for error_type, hint in hints.items():
        if isinstance(error, error_type):
            return hint
    
    return None
