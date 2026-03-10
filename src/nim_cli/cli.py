"""
Main CLI application for NIM CLI.

This module provides the main application class that:
- Initializes all components
- Runs the main event loop
- Handles user interaction
- Manages conversation state
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from nim_cli import __version__
from nim_cli.core.config import Config, ConfigManager, get_config, get_config_manager
from nim_cli.core.client import NIMClient, ChatResponse
from nim_cli.core.errors import (
    NIMCLIError,
    NIMRateLimitError,
    NIMTimeoutError,
    NIMAuthenticationError,
    CircuitOpenError,
    get_error_hint,
)
from nim_cli.core.metrics import get_metrics
from nim_cli.core.retry import CircuitBreaker, RateLimitHandler
from nim_cli.core.streaming import StreamChunk
from nim_cli.ui.theme import Theme, get_theme, set_theme
from nim_cli.ui.renderer import (
    MessageRenderer,
    MessageStats,
    StreamingDisplay,
    create_console,
)
from nim_cli.ui.input import (
    InputHandler,
    InputResult,
    InputState,
    create_input_handler,
)
from nim_cli.ui.animations import get_spinner_names

if TYPE_CHECKING:
    pass


# =============================================================================
# Application State
# =============================================================================


@dataclass
class ConversationMessage:
    """A message in the conversation."""
    
    role: str  # "user", "assistant", "system"
    content: str
    tokens: int = 0
    timestamp: float = 0.0


@dataclass
class ConversationState:
    """State of the conversation."""
    
    messages: list[ConversationMessage] = field(default_factory=list)
    total_tokens: int = 0
    turn_count: int = 0
    
    def add_message(
        self,
        role: str,
        content: str,
        tokens: int = 0,
    ) -> None:
        """Add a message to the conversation."""
        import time
        self.messages.append(
            ConversationMessage(
                role=role,
                content=content,
                tokens=tokens,
                timestamp=time.time(),
            )
        )
        self.total_tokens += tokens
        if role == "user":
            self.turn_count += 1
    
    def clear(self) -> None:
        """Clear the conversation."""
        self.messages.clear()
        self.total_tokens = 0
        self.turn_count = 0
    
    def get_context(self) -> list[dict]:
        """Get messages for API context."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]


# =============================================================================
# Main Application
# =============================================================================


class Application:
    """
    Main CLI application.
    
    This class orchestrates all components and manages the main
    event loop for the CLI.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        theme: Optional[str] = None,
    ) -> None:
        """
        Initialize application.
        
        Args:
            config: Configuration (uses global if not provided)
            theme: Theme name to use
        """
        # Load configuration
        self._config = config or get_config()
        
        # Set theme
        if theme:
            set_theme(theme)
        self._theme = get_theme()
        
        # Create console
        self._console = create_console()
        
        # Create components
        self._input_handler = create_input_handler()
        self._message_renderer = MessageRenderer(self._console)
        self._streaming_display: Optional[StreamingDisplay] = None
        
        # Client (lazy initialization)
        self._client: Optional[NIMClient] = None
        
        # State
        self._conversation = ConversationState()
        self._running = False
        self._debug = False
    
    @property
    def config(self) -> Config:
        """Get configuration."""
        return self._config
    
    @property
    def client(self) -> NIMClient:
        """Get API client (lazy init)."""
        if self._client is None:
            self._client = NIMClient(self._config)
        return self._client
    
    def run(self) -> int:
        """
        Run the application.
        
        Returns:
            Exit code
        """
        try:
            # Setup signal handlers
            self._setup_signals()
            
            # Show welcome
            self._show_welcome()
            
            # Run main loop
            self._running = True
            asyncio.run(self._main_loop())
            
            return 0
            
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            self._console.print(f"\nFatal error: {e}", style="error")
            return 1
        finally:
            self._cleanup()
    
    def _setup_signals(self) -> None:
        """Setup signal handlers."""
        def handle_sigint(signum, frame):
            if self._running:
                self._console.print("\nInterrupted. Press Ctrl+C again to exit.")
                self._running = False
            else:
                sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_sigint)
    
    def _cleanup(self) -> None:
        """Cleanup resources (async close is handled inside _main_loop's finally)."""
        pass
    
    def _show_welcome(self) -> None:
        """Show welcome message."""
        # Create title
        title = Text()
        title.append("NIM CLI ", style=f"bold {self._theme.colors.primary.name}")
        title.append(f"v{__version__}", style=self._theme.colors.text_muted.name)
        
        # Create subtitle
        subtitle = Text()
        subtitle.append("Powered by NVIDIA NIM", style=self._theme.colors.secondary.name)
        
        # Show model info
        model_info = Text()
        model_info.append(f"Model: ", style=self._theme.colors.text_muted.name)
        model_info.append(self._config.model.name, style=self._theme.colors.accent.name)
        
        # Create panel
        content = Text()
        content.append("Type your message and press Enter to send.\n")
        content.append("Type /help for available commands.\n")
        content.append("Press Ctrl+D to exit.\n\n")
        content.append(model_info)
        
        panel = Panel(
            content,
            title=title,
            border_style=Style(color=self._theme.colors.primary.name),
            padding=(1, 2),
        )
        
        self._console.print(panel)
        self._console.print()
    
    async def _main_loop(self) -> None:
        """Main event loop."""
        try:
            while self._running:
                try:
                    # Get input
                    result = await self._input_handler.get_input()

                    # Handle empty input
                    if result.is_empty:
                        continue

                    # Handle commands
                    if result.is_command:
                        should_continue = await self._handle_command(result)
                        if not should_continue:
                            break
                        continue

                    # Handle message
                    await self._handle_message(result.text)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self._handle_error(e)
        finally:
            # Must close the async client inside the running event loop
            if self._client:
                await self._client.close()
    
    async def _handle_command(self, result: InputResult) -> bool:
        """
        Handle a command.
        
        Args:
            result: Parsed command result
            
        Returns:
            True if should continue, False to exit
        """
        cmd = result.command
        args = result.args
        
        if cmd in ("exit", "quit", "q"):
            self._console.print("\nGoodbye!", style=self._theme.colors.text_muted.name)
            return False
        
        elif cmd in ("help", "h", "?"):
            self._show_help(args[0] if args else None)
        
        elif cmd in ("clear", "cls", "c"):
            self._conversation.clear()
            self._console.clear()
            self._show_welcome()
        
        elif cmd == "model":
            if args:
                self._config.model.name = args[0].lower()
                get_config_manager().save(self._config)
                self._console.print(f"Model: {self._config.model.name}", style=self._theme.colors.success.name)
            else:
                self._console.print(f"Current model: {self._config.model.name}")
        
        elif cmd in ("models", "ml"):
            self._show_models()
        
        elif cmd == "config":
            self._show_config(args)
        
        elif cmd == "theme":
            if args:
                try:
                    set_theme(args[0])
                    self._theme = get_theme()
                    self._config.display.theme = args[0].lower()
                    get_config_manager().save(self._config)
                    self._console.print(f"Theme: {args[0]}", style=self._theme.colors.success.name)
                except ValueError as e:
                    self._console.print(f"Error: {e}", style="error")
            else:
                self._console.print(f"Current theme: {self._theme.name}")
        
        elif cmd == "debug":
            self._debug = not self._debug
            self._console.print(f"Debug mode: {'on' if self._debug else 'off'}")

        elif cmd == "stats":
            self._show_stats()

        elif cmd in ("history", "hist"):
            self._show_history()

        elif cmd == "save":
            await self._save_conversation(args[0] if args else None)

        elif cmd == "load":
            await self._load_conversation(args[0] if args else None)

        elif cmd == "copy":
            self._copy_last_response()

        else:
            self._console.print(f"Unknown command: /{cmd}", style="error")
        
        return True
    
    def _show_help(self, command: Optional[str] = None) -> None:
        """Show help message."""
        if command:
            # Show help for specific command
            cmd = self._input_handler.registry.get(command)
            if cmd:
                text = Text()
                text.append(f"/{cmd.name}", style=self._theme.colors.primary)
                if cmd.aliases:
                    text.append(f" (aliases: {', '.join(cmd.aliases)})")
                text.append(f"\n{cmd.description}")
                if cmd.usage:
                    text.append(f"\nUsage: {cmd.usage}")
                self._console.print(Panel(text, border_style=Style(color=self._theme.colors.primary.name)))
            else:
                self._console.print(f"Unknown command: {command}")
            return
        
        # Show all commands
        commands = self._input_handler.registry.list_commands()
        
        content = Text()
        for cmd in sorted(commands, key=lambda c: c.name):
            content.append(f"/{cmd.name}", style=self._theme.colors.primary.name)
            if cmd.aliases:
                content.append(f" ({', '.join(cmd.aliases)})")
            content.append(f" - {cmd.description}\n")
        
        panel = Panel(
            content,
            title="Available Commands",
            border_style=Style(color=self._theme.colors.primary.name),
        )
        
        self._console.print(panel)
    
    def _show_models(self) -> None:
        """Show available models."""
        models = [
            ("meta/llama-3.1-8b-instruct", "Llama 3.1 8B (fast)"),
            ("meta/llama-3.1-70b-instruct", "Llama 3.1 70B (balanced)"),
            ("meta/llama-3.1-405b-instruct", "Llama 3.1 405B (powerful)"),
            ("meta/llama-3.2-1b-instruct", "Llama 3.2 1B (ultra fast)"),
            ("meta/llama-3.2-3b-instruct", "Llama 3.2 3B (fast)"),
            ("nvidia/llama-3.1-nemotron-70b-instruct", "Nemotron 70B"),
            ("deepseek-ai/deepseek-r1", "DeepSeek R1"),
            ("qwen/qwen2.5-7b-instruct", "Qwen 2.5 7B"),
            ("mistralai/mistral-large", "Mistral Large"),
            ("mistralai/mixtral-8x7b-instruct", "Mixtral 8x7B"),
        ]
        
        content = Text()
        for model_id, description in models:
            if model_id == self._config.model.name:
                content.append("● ", style=self._theme.colors.success.name)
            else:
                content.append("○ ", style=self._theme.colors.text_muted.name)
            content.append(model_id, style=self._theme.colors.primary.name)
            content.append(f" - {description}\n")
        
        panel = Panel(
            content,
            title="Available Models",
            border_style=Style(color=self._theme.colors.primary.name),
        )
        
        self._console.print(panel)
    
    def _show_config(self, args: list[str]) -> None:
        """Show or modify configuration."""
        if not args:
            # Show all config
            content = Text()
            content.append(f"Model: {self._config.model.name}\n")
            content.append(f"Temperature: {self._config.model.temperature}\n")
            content.append(f"Max Tokens: {self._config.model.max_tokens}\n")
            content.append(f"Theme: {self._theme.name}\n")
            content.append(f"Stream: {self._config.model.stream}\n")

            panel = Panel(
                content,
                title="Configuration",
                border_style=Style(color=self._theme.colors.primary.name),
            )
            self._console.print(panel)

        elif len(args) == 1:
            # Show specific key
            key = args[0]
            key_map = {
                "model": lambda: self._config.model.name,
                "temperature": lambda: str(self._config.model.temperature),
                "max_tokens": lambda: str(self._config.model.max_tokens),
                "theme": lambda: self._theme.name,
            }
            if key in key_map:
                self._console.print(f"{key}: {key_map[key]()}")
            else:
                self._console.print(f"Unknown config key: {key}")

        elif len(args) == 2:
            # Set config key = value
            key, value = args[0], args[1]
            try:
                if key == "model":
                    self._config.model.name = value.lower()
                elif key == "temperature":
                    self._config.model.temperature = float(value)
                elif key == "max_tokens":
                    self._config.model.max_tokens = int(value)
                elif key == "theme":
                    set_theme(value)
                    self._theme = get_theme()
                    self._config.display.theme = value.lower()
                else:
                    self._console.print(f"Unknown config key: {key}", style="error")
                    return
                get_config_manager().save(self._config)
                self._console.print(f"Set {key} = {value}", style=self._theme.colors.success.name)
            except (ValueError, TypeError) as e:
                self._console.print(f"Invalid value for {key}: {e}", style="error")
    
    def _show_stats(self) -> None:
        """Show session statistics."""
        metrics = get_metrics()
        summary = metrics.get_summary()
        
        content = Text()
        content.append("Session Statistics\n\n", style="bold")
        content.append(f"Duration: {summary['session']['duration_seconds']:.1f}s\n")
        content.append(f"Total Requests: {summary['api']['total_requests']}\n")
        content.append(f"Success Rate: {summary['api']['success_rate']*100:.1f}%\n")
        content.append(f"Total Tokens: {summary['api']['total_tokens_received']}\n")
        content.append(f"Avg Latency: {summary['api']['avg_latency_ms']:.0f}ms\n")
        content.append(f"Avg TTFT: {summary['api']['avg_ttft_ms']:.0f}ms\n")
        content.append(f"Throughput: {summary['api']['avg_throughput_tps']:.1f} tok/s\n")
        
        if summary['errors']['total'] > 0:
            content.append(f"\nErrors: {summary['errors']['total']}\n", style="warning")
        
        panel = Panel(
            content,
            title="Statistics",
            border_style=Style(color=self._theme.colors.primary.name),
        )
        
        self._console.print(panel)
    
    async def _handle_message(self, message: str) -> None:
        """Handle a user message."""
        # Show user message
        user_render = self._message_renderer.render_user(message)
        self._console.print(user_render)
        self._console.print()
        
        # Add to conversation
        self._conversation.add_message("user", message)
        
        # Get response
        try:
            if self._config.model.stream:
                await self._stream_response()
            else:
                await self._get_complete_response()
                
        except Exception as e:
            self._handle_error(e)
    
    async def _stream_response(self) -> None:
        """Stream response from API."""
        display = StreamingDisplay(self._console)

        try:
            display.start()

            async for chunk in self.client.chat_stream(
                self._conversation.messages[-1].content,
                history=self._conversation.get_context()[:-1],
            ):
                display.update(chunk)

            content = display.finish()

            self._conversation.add_message(
                "assistant",
                content,
                tokens=display._token_count,
            )

        except Exception:
            display.cancel()
            raise
    
    async def _get_complete_response(self) -> None:
        """Get complete response from API."""
        response = await self.client.chat(
            self._conversation.messages[-1].content,
            history=self._conversation.get_context()[:-1],
        )

        self._conversation.add_message(
            "assistant",
            response.content,
            tokens=response.total_tokens,
        )

        stats = MessageStats(
            tokens=response.total_tokens,
            duration_ms=response.latency_ms,
        )

        render = self._message_renderer.render_assistant(
            response.content,
            stats=stats,
        )

        self._console.print(render)
        self._console.print()
    
    def _handle_error(self, error: Exception) -> None:
        """Handle an error."""
        # Get hint
        hint = get_error_hint(error)
        
        # Special handling for specific errors
        if isinstance(error, NIMRateLimitError):
            retry_after = error.details.get("retry_after_seconds", 60)
            hint = f"Rate limited. Wait {retry_after}s before retrying."
        
        elif isinstance(error, NIMAuthenticationError):
            hint = "Check your NVIDIA_API_KEY environment variable."
        
        elif isinstance(error, CircuitOpenError):
            hint = "API is experiencing issues. Wait a moment and try again."
        
        # Render error
        error_render = self._message_renderer.render_error(error, hint)
        self._console.print(error_render)
        self._console.print()


    # =========================================================================
    # History / Save / Load / Copy
    # =========================================================================

    def _show_history(self) -> None:
        """Show conversation history summary."""
        if not self._conversation.messages:
            self._console.print("No conversation history.", style=self._theme.colors.text_muted.name)
            return

        content = Text()
        for i, msg in enumerate(self._conversation.messages):
            role_color = {
                "user": self._theme.colors.user_message.name,
                "assistant": self._theme.colors.assistant_message.name,
                "system": self._theme.colors.system_message.name,
            }.get(msg.role, self._theme.colors.text_muted.name)
            content.append(f"[{i + 1}] {msg.role.capitalize()}: ", style=f"bold {role_color}")
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            content.append(f"{preview}\n")

        panel = Panel(
            content,
            title=f"History -- {len(self._conversation.messages)} messages, {self._conversation.total_tokens} tokens",
            border_style=Style(color=self._theme.colors.primary.name),
        )
        self._console.print(panel)

    async def _save_conversation(self, filename: Optional[str] = None) -> None:
        """Save conversation to a JSON file."""
        if not self._conversation.messages:
            self._console.print("Nothing to save.", style=self._theme.colors.text_muted.name)
            return

        if filename is None:
            filename = f"nim_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path = Path(filename)
        data = {
            "model": self._config.model.name,
            "messages": [
                {"role": msg.role, "content": msg.content, "tokens": msg.tokens}
                for msg in self._conversation.messages
            ],
            "total_tokens": self._conversation.total_tokens,
            "turns": self._conversation.turn_count,
        }
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._console.print(f"Saved -> {path.resolve()}", style=self._theme.colors.success.name)
        except Exception as e:
            self._console.print(f"Save failed: {e}", style="error")

    async def _load_conversation(self, filename: Optional[str] = None) -> None:
        """Load conversation from a JSON file."""
        if filename is None:
            self._console.print("Usage: /load <filename>", style=self._theme.colors.text_muted.name)
            return

        path = Path(filename)
        if not path.exists():
            self._console.print(f"File not found: {filename}", style="error")
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._conversation.clear()
            for msg in data.get("messages", []):
                self._conversation.add_message(msg["role"], msg["content"], msg.get("tokens", 0))
            self._console.print(
                f"Loaded {len(self._conversation.messages)} messages from {path.name}",
                style=self._theme.colors.success.name,
            )
        except Exception as e:
            self._console.print(f"Load failed: {e}", style="error")

    def _copy_last_response(self) -> None:
        """Copy last assistant response to clipboard."""
        last = next(
            (m for m in reversed(self._conversation.messages) if m.role == "assistant"),
            None,
        )
        if not last:
            self._console.print("No response to copy.", style=self._theme.colors.text_muted.name)
            return

        try:
            if sys.platform == "win32":
                subprocess.run(["clip"], input=last.content.encode("utf-16"), check=True, capture_output=True)
            elif sys.platform == "darwin":
                subprocess.run(["pbcopy"], input=last.content.encode(), check=True, capture_output=True)
            else:
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=last.content.encode(), check=True, capture_output=True,
                    )
                except FileNotFoundError:
                    subprocess.run(
                        ["xsel", "--clipboard", "--input"],
                        input=last.content.encode(), check=True, capture_output=True,
                    )
            self._console.print("Copied to clipboard!", style=self._theme.colors.success.name)
        except Exception as e:
            self._console.print(f"Copy failed: {e}", style="error")


# =============================================================================
# Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    app = Application()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
