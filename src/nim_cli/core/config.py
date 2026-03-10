"""
Configuration management for NIM CLI.

This module provides a comprehensive configuration system that:
- Loads configuration from multiple sources with priority
- Validates all settings with Pydantic
- Provides secure API key handling
- Supports multiple profiles

Configuration Sources (in order of priority):
1. CLI arguments (highest)
2. Environment variables (NIM_CLI_*)
3. User config file (~/.config/nim-cli/config.toml)
4. System config file (/etc/nim-cli/config.toml)
5. Built-in defaults (lowest)
"""

from __future__ import annotations

import os
import platform
import stat
from pathlib import Path
from typing import Any, Literal, Optional

import orjson
import tomli_w
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

from nim_cli.core.errors import (
    APIKeyError,
    ConfigError,
    ConfigFileError,
    ConfigValidationError,
)


# =============================================================================
# Configuration Models
# =============================================================================


class ModelConfig(BaseModel):
    """Configuration for the AI model."""
    
    name: str = Field(
        default="meta/llama-3.1-8b-instruct",
        description="Model identifier for NVIDIA NIM",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=131072,
        description="Maximum tokens in response",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Top-p sampling (0-1)",
    )
    top_k: int = Field(
        default=40,
        ge=0,
        description="Top-k sampling (0 = disabled)",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty (-2 to 2)",
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty (-2 to 2)",
    )
    stream: bool = Field(
        default=True,
        description="Enable streaming responses",
    )
    
    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model name follows expected format."""
        if not v:
            raise ValueError("Model name cannot be empty")
        # Normalize model name
        v = v.strip().lower()
        return v


class NetworkConfig(BaseModel):
    """Configuration for network/API connections."""
    
    base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM API base URL",
    )
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Request timeout in seconds",
    )
    connect_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Connection timeout in seconds",
    )
    read_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Read timeout for streaming chunks",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )
    retry_base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Base delay for exponential backoff",
    )
    retry_max_delay: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay between retries",
    )
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="HTTP connection pool size",
    )
    keepalive_expiry: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Keep-alive expiry in seconds",
    )


class DisplayConfig(BaseModel):
    """Configuration for UI/display."""
    
    theme: str = Field(
        default="dark",
        description="UI theme name",
    )
    animation_speed: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Animation speed multiplier",
    )
    show_token_count: bool = Field(
        default=True,
        description="Show token counts",
    )
    show_timing: bool = Field(
        default=True,
        description="Show timing information",
    )
    show_model_name: bool = Field(
        default=True,
        description="Show model name in header",
    )
    markdown_enabled: bool = Field(
        default=True,
        description="Render markdown in responses",
    )
    code_theme: str = Field(
        default="monokai",
        description="Syntax highlighting theme for code",
    )
    max_line_width: int = Field(
        default=120,
        ge=40,
        le=500,
        description="Maximum line width for wrapping",
    )
    streaming_typing_speed: int = Field(
        default=0,  # 0 = instant, higher = slower
        ge=0,
        le=100,
        description="Typing animation delay (ms per char)",
    )
    spinner_style: str = Field(
        default="dots",
        description="Spinner animation style",
    )
    
    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        """Validate theme name."""
        valid_themes = {"dark", "light", "monokai", "dracula", "nord", "solarized"}
        v = v.lower().strip()
        if v not in valid_themes:
            # Don't fail, just use default
            return "dark"
        return v


class BehaviorConfig(BaseModel):
    """Configuration for agent behavior."""
    
    auto_save_history: bool = Field(
        default=True,
        description="Auto-save conversation history",
    )
    history_limit: int = Field(
        default=100,
        ge=0,
        le=10000,
        description="Maximum history entries to keep",
    )
    context_window: int = Field(
        default=128000,
        ge=1024,
        le=1000000,
        description="Context window size in tokens",
    )
    auto_compact: bool = Field(
        default=True,
        description="Auto-compact context when approaching limit",
    )
    compact_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Compact when context reaches this fraction",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt",
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="WARNING",
        description="Log level",
    )
    file: Optional[str] = Field(
        default=None,
        description="Log file path (None = no file logging)",
    )
    format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Log format string",
    )
    include_timing: bool = Field(
        default=True,
        description="Include timing information in logs",
    )


class Config(BaseModel):
    """Root configuration model."""
    
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="NVIDIA API key",
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model settings",
    )
    network: NetworkConfig = Field(
        default_factory=NetworkConfig,
        description="Network settings",
    )
    display: DisplayConfig = Field(
        default_factory=DisplayConfig,
        description="Display settings",
    )
    behavior: BehaviorConfig = Field(
        default_factory=BehaviorConfig,
        description="Behavior settings",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging settings",
    )
    profile: str = Field(
        default="default",
        description="Current profile name",
    )
    
    @model_validator(mode="after")
    def validate_api_key(self) -> "Config":
        """Ensure API key is available when needed."""
        # API key will be loaded from environment if not in config
        return self
    
    def get_api_key(self) -> str:
        """
        Get the API key, raising an error if not available.
        
        Returns:
            The API key string
            
        Raises:
            APIKeyError: If API key is not configured
        """
        # Try config first
        if self.api_key:
            return self.api_key.get_secret_value()
        
        # Try environment variable
        env_key = os.environ.get("NVIDIA_API_KEY")
        if env_key:
            return env_key
        
        raise APIKeyError()


# =============================================================================
# Configuration Manager
# =============================================================================


class ConfigManager:
    """
    Manages configuration loading, validation, and saving.
    
    This class implements a layered configuration system where values from
    higher-priority sources override those from lower-priority sources.
    """
    
    # Environment variable prefix
    ENV_PREFIX = "NIM_CLI_"
    
    # Default config paths
    DEFAULT_CONFIG_DIR = Path.home() / ".config" / "nim-cli"
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.toml"
    
    # System config path
    SYSTEM_CONFIG_FILE = Path("/etc/nim-cli/config.toml")
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        profile: Optional[str] = None,
    ) -> None:
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional custom config file path
            profile: Optional profile name to use
        """
        self._config_path = config_path or self.DEFAULT_CONFIG_FILE
        self._profile = profile or "default"
        self._config: Optional[Config] = None
        self._overrides: dict[str, Any] = {}
    
    @property
    def config(self) -> Config:
        """Get the loaded configuration, loading if necessary."""
        if self._config is None:
            self._config = self._load()
        return self._config
    
    @property
    def config_path(self) -> Path:
        """Get the config file path."""
        return self._config_path
    
    def reload(self) -> Config:
        """Reload configuration from all sources."""
        self._config = None
        return self.config
    
    def set_override(self, key: str, value: Any) -> None:
        """
        Set a configuration override.
        
        Overrides have the highest priority and will be used even if
        they differ from file or environment configuration.
        
        Args:
            key: Dot-notation key (e.g., "model.temperature")
            value: The value to set
        """
        self._overrides[key] = value
        self._config = None  # Force reload
    
    def save(self, config: Optional[Config] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current if None)
            
        Raises:
            ConfigError: If saving fails
        """
        config = config or self.config
        
        try:
            # Ensure directory exists
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict, handling SecretStr
            config_dict = self._config_to_dict(config)
            
            # Write to file
            with open(self._config_path, "wb") as f:
                tomli_w.dump(config_dict, f)
            
            # Set secure permissions (owner read/write only)
            self._config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            
        except PermissionError as e:
            raise ConfigError(
                f"Permission denied writing config: {e}",
                config_path=str(self._config_path),
            )
        except Exception as e:
            raise ConfigError(
                f"Failed to save config: {e}",
                config_path=str(self._config_path),
            )
    
    def _load(self) -> Config:
        """
        Load configuration from all sources.
        
        Returns:
            Loaded and validated configuration
        """
        # Start with defaults
        config_dict: dict[str, Any] = {}
        
        # Layer 4: System config file
        if self.SYSTEM_CONFIG_FILE.exists():
            config_dict = self._deep_merge(
                config_dict,
                self._load_file(self.SYSTEM_CONFIG_FILE)
            )
        
        # Layer 3: User config file
        if self._config_path.exists():
            file_config = self._load_file(self._config_path)
            # Handle profiles
            if "profiles" in file_config and self._profile in file_config["profiles"]:
                profile_config = file_config["profiles"][self._profile]
                file_config = self._deep_merge(file_config, profile_config)
                file_config.pop("profiles", None)
            config_dict = self._deep_merge(config_dict, file_config)
        
        # Layer 2: Environment variables
        env_config = self._load_env()
        config_dict = self._deep_merge(config_dict, env_config)
        
        # Layer 1: Overrides (highest priority)
        config_dict = self._apply_overrides(config_dict, self._overrides)
        
        # Handle API key specially (from NVIDIA_API_KEY env var)
        if "api_key" not in config_dict:
            env_api_key = os.environ.get("NVIDIA_API_KEY")
            if env_api_key:
                config_dict["api_key"] = env_api_key
        
        # Create and validate config
        try:
            return Config(**config_dict)
        except Exception as e:
            raise ConfigValidationError(
                f"Invalid configuration: {e}",
                details={"raw_error": str(e)},
            )
    
    def _load_file(self, path: Path) -> dict[str, Any]:
        """Load configuration from a TOML file."""
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            return {}
        except tomllib.TOMLDecodeError as e:
            raise ConfigFileError(
                f"Invalid TOML in config file: {e}",
                config_path=str(path),
                line=getattr(e, "lineno", None),
            )
        except Exception as e:
            raise ConfigFileError(
                f"Failed to read config file: {e}",
                config_path=str(path),
            )
    
    def _load_env(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        config: dict[str, Any] = {}
        
        # Map of env vars to config keys
        env_mappings = {
            f"{self.ENV_PREFIX}API_KEY": "api_key",
            f"{self.ENV_PREFIX}MODEL": "model.name",
            f"{self.ENV_PREFIX}TEMPERATURE": "model.temperature",
            f"{self.ENV_PREFIX}MAX_TOKENS": "model.max_tokens",
            f"{self.ENV_PREFIX}THEME": "display.theme",
            f"{self.ENV_PREFIX}LOG_LEVEL": "logging.level",
            f"{self.ENV_PREFIX}BASE_URL": "network.base_url",
            f"{self.ENV_PREFIX}TIMEOUT": "network.timeout",
        }
        
        for env_key, config_key in env_mappings.items():
            value = os.environ.get(env_key)
            if value is not None:
                self._set_nested(config, config_key.split("."), value)
        
        return config
    
    def _apply_overrides(
        self,
        config: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply override values to configuration."""
        config = config.copy()
        for key, value in overrides.items():
            self._set_nested(config, key.split("."), value)
        return config
    
    def _set_nested(
        self,
        d: dict[str, Any],
        keys: list[str],
        value: Any,
    ) -> None:
        """Set a nested dictionary value."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        # Try to parse as JSON for complex values
        if isinstance(value, str):
            try:
                value = orjson.loads(value)
            except (orjson.JSONDecodeError, ValueError):
                pass
        d[keys[-1]] = value
    
    def _deep_merge(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _config_to_dict(self, config: Config) -> dict[str, Any]:
        """Convert Config to a dictionary suitable for TOML serialization."""
        def convert_value(v: Any) -> Any:
            if isinstance(v, SecretStr):
                return v.get_secret_value()
            elif isinstance(v, BaseModel):
                return {k: convert_value(val) for k, val in v.model_dump().items()}
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(item) for item in v]
            return v
        
        return {k: convert_value(v) for k, v in config.model_dump().items()}


# =============================================================================
# Module-level convenience
# =============================================================================

# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Config:
    """Get the current configuration."""
    return get_config_manager().config


def ensure_config_dir() -> Path:
    """Ensure configuration directory exists and return its path."""
    config_dir = ConfigManager.DEFAULT_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    # Set secure permissions on config directory
    config_dir.chmod(stat.S_IRWXU)
    return config_dir
