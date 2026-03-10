"""
NIM CLI - World-class agentic CLI powered by NVIDIA NIM.

A lightweight, robust, and beautiful terminal interface for AI agents.
"""

__version__ = "1.0.0"
__author__ = "NIM CLI Team"
__license__ = "MIT"

from nim_cli.core.errors import (
    NIMCLIError,
    ConfigError,
    APIKeyError,
    NIMConnectionError,
    NIMTimeoutError,
    NIMRateLimitError,
    NIMAuthenticationError,
    NIMContentFilterError,
    StreamError,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "NIMCLIError",
    "ConfigError",
    "APIKeyError",
    "NIMConnectionError",
    "NIMTimeoutError",
    "NIMRateLimitError",
    "NIMAuthenticationError",
    "NIMContentFilterError",
    "StreamError",
]
