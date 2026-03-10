"""
Input handling for NIM CLI.

This module provides:
- Multi-line input support
- Command handling
- History management
- Keyboard shortcuts
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.validation import Validator
from prompt_toolkit.widgets import TextArea

from nim_cli.core.config import get_config
from nim_cli.core.errors import CommandError, InputError
from nim_cli.ui.theme import get_theme


# =============================================================================
# Input State
# =============================================================================


class InputState(Enum):
    """States for input handling."""
    IDLE = auto()
    TYPING = auto()
    MULTI_LINE = auto()
    SENDING = auto()
    STREAMING = auto()


@dataclass
class InputResult:
    """Result of input processing."""
    
    text: str
    is_command: bool = False
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    
    @property
    def is_empty(self) -> bool:
        """Check if input is empty."""
        return not self.text.strip()


# =============================================================================
# Command Registry
# =============================================================================


@dataclass
class Command:
    """Represents a CLI command."""
    
    name: str
    description: str
    aliases: list[str] = field(default_factory=list)
    usage: Optional[str] = None
    handler: Optional[Callable[[list[str]], None]] = None
    
    def matches(self, text: str) -> bool:
        """Check if text matches this command."""
        if text.startswith(f"/{self.name}"):
            return True
        for alias in self.aliases:
            if text.startswith(f"/{alias}"):
                return True
        return False


class CommandRegistry:
    """Registry for CLI commands."""
    
    def __init__(self) -> None:
        """Initialize registry."""
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}
        self._register_builtin_commands()
    
    def _register_builtin_commands(self) -> None:
        """Register built-in commands."""
        builtin_commands = [
            Command(
                name="help",
                description="Show available commands",
                aliases=["h", "?"],
                usage="/help [command]",
            ),
            Command(
                name="exit",
                description="Exit the CLI",
                aliases=["quit", "q"],
            ),
            Command(
                name="clear",
                description="Clear the conversation",
                aliases=["cls", "c"],
            ),
            Command(
                name="model",
                description="Switch or show current model",
                aliases=["m"],
                usage="/model [name]",
            ),
            Command(
                name="models",
                description="List available models",
                aliases=["ml"],
            ),
            Command(
                name="config",
                description="View or modify configuration",
                usage="/config [key] [value]",
            ),
            Command(
                name="history",
                description="Show conversation history",
                aliases=["hist"],
            ),
            Command(
                name="save",
                description="Save conversation to file",
                usage="/save [filename]",
            ),
            Command(
                name="load",
                description="Load conversation from file",
                usage="/load [filename]",
            ),
            Command(
                name="theme",
                description="Change or show theme",
                usage="/theme [name]",
            ),
            Command(
                name="debug",
                description="Toggle debug mode",
            ),
            Command(
                name="stats",
                description="Show session statistics",
            ),
            Command(
                name="copy",
                description="Copy last response to clipboard",
            ),
        ]
        
        for cmd in builtin_commands:
            self.register(cmd)
    
    def register(self, command: Command) -> None:
        """Register a command."""
        self._commands[command.name] = command
        
        # Register aliases
        for alias in command.aliases:
            self._aliases[alias] = command.name
    
    def get(self, name: str) -> Optional[Command]:
        """Get a command by name or alias."""
        # Try direct name
        if name in self._commands:
            return self._commands[name]
        
        # Try alias
        if name in self._aliases:
            return self._commands.get(self._aliases[name])
        
        return None
    
    def parse(self, text: str) -> InputResult:
        """
        Parse input text.
        
        Args:
            text: Input text
            
        Returns:
            InputResult with parsed command or text
        """
        text = text.strip()
        
        if not text:
            return InputResult(text="")
        
        if not text.startswith("/"):
            return InputResult(text=text, is_command=False)
        
        # Parse command
        parts = text[1:].split(maxsplit=1)
        if not parts:
            return InputResult(text=text, is_command=False)
        
        cmd_name = parts[0].lower()
        args = parts[1].split() if len(parts) > 1 else []
        
        return InputResult(
            text=text,
            is_command=True,
            command=cmd_name,
            args=args,
        )
    
    def list_commands(self) -> list[Command]:
        """List all commands."""
        return list(self._commands.values())
    
    def get_completions(self, text: str) -> list[str]:
        """Get command completions for text."""
        if not text.startswith("/"):
            return []
        
        prefix = text[1:].lower()
        completions = []
        
        for name, cmd in self._commands.items():
            if name.startswith(prefix):
                completions.append(f"/{name}")
            for alias in cmd.aliases:
                if alias.startswith(prefix):
                    completions.append(f"/{alias}")
        
        return completions


# =============================================================================
# Input Handler
# =============================================================================


class InputHandler:
    """
    Handles user input with advanced features.
    
    This class provides:
    - Multi-line input
    - Command parsing
    - History management
    - Keyboard shortcuts
    """
    
    def __init__(
        self,
        history_file: Optional[Path] = None,
    ) -> None:
        """
        Initialize input handler.
        
        Args:
            history_file: Path to history file
        """
        self._theme = get_theme()
        self._config = get_config()
        self._registry = CommandRegistry()
        self._state = InputState.IDLE
        
        # Setup history
        if history_file:
            history_dir = history_file.parent
            history_dir.mkdir(parents=True, exist_ok=True)
            self._history = FileHistory(str(history_file))
        else:
            self._history = InMemoryHistory()
        
        # Setup session
        self._session: Optional[PromptSession] = None
        self._key_bindings = self._create_key_bindings()
        
        # State
        self._multi_line_buffer: list[str] = []
        self._last_input: Optional[str] = None
    
    @property
    def state(self) -> InputState:
        """Get current input state."""
        return self._state
    
    @property
    def registry(self) -> CommandRegistry:
        """Get command registry."""
        return self._registry
    
    def _create_key_bindings(self) -> KeyBindings:
        """Create key bindings."""
        kb = KeyBindings()
        
        @kb.add(Keys.Enter, filter=Condition(lambda: self._state == InputState.IDLE))
        def _(event) -> None:
            """Handle Enter in normal mode."""
            buffer = event.app.current_buffer
            
            # Check for multi-line mode trigger
            text = buffer.text
            if text.endswith("\\"):
                # Enter multi-line mode
                self._state = InputState.MULTI_LINE
                self._multi_line_buffer = [text[:-1]]  # Remove backslash
                buffer.text = ""
                return
            
            # Normal submit
            event.current_buffer.validate_and_handle()
        
        @kb.add(Keys.Enter, filter=Condition(lambda: self._state == InputState.MULTI_LINE))
        def _(event) -> None:
            """Handle Enter in multi-line mode."""
            buffer = event.app.current_buffer
            text = buffer.text
            
            # Empty line ends multi-line mode
            if not text.strip():
                self._state = InputState.IDLE
                buffer.text = "\n".join(self._multi_line_buffer)
                self._multi_line_buffer = []
                event.current_buffer.validate_and_handle()
                return
            
            # Check for explicit end
            if text.strip() == "```":
                self._multi_line_buffer.append(text)
                self._state = InputState.IDLE
                buffer.text = "\n".join(self._multi_line_buffer)
                self._multi_line_buffer = []
                event.current_buffer.validate_and_handle()
                return
            
            # Add to buffer
            self._multi_line_buffer.append(text)
            buffer.text = ""
        
        @kb.add(Keys.Escape, filter=Condition(lambda: self._state == InputState.MULTI_LINE))
        def _(event) -> None:
            """Cancel multi-line mode."""
            self._state = InputState.IDLE
            self._multi_line_buffer = []
            event.app.current_buffer.text = ""
        
        @kb.add(Keys.ControlC)
        def _(event) -> None:
            """Handle Ctrl+C."""
            if self._state == InputState.MULTI_LINE:
                self._state = InputState.IDLE
                self._multi_line_buffer = []
                event.app.current_buffer.text = ""
            else:
                event.app.exit(exception=KeyboardInterrupt())
        
        @kb.add(Keys.ControlD)
        def _(event) -> None:
            """Handle Ctrl+D - exit."""
            buffer = event.app.current_buffer
            if buffer.text:
                buffer.delete(0)  # Delete character like shell
            else:
                event.app.exit(exception=EOFError())
        
        @kb.add(Keys.ControlL)
        def _(event) -> None:
            """Handle Ctrl+L - clear screen."""
            event.app.renderer.clear()
        
        @kb.add(Keys.Tab)
        def _(event) -> None:
            """Handle Tab - autocomplete."""
            buffer = event.app.current_buffer
            text = buffer.text
            
            if text.startswith("/"):
                completions = self._registry.get_completions(text)
                if completions:
                    buffer.text = completions[0] + " "
                    buffer.cursor_position = len(buffer.text)
        
        return kb
    
    def _create_prompt_style(self) -> PromptStyle:
        """Create prompt styling."""
        theme = self._theme
        
        return PromptStyle.from_dict({
            "prompt": f"bold fg:{theme.colors.primary}",
            "": f"fg:{theme.colors.text}",
        })
    
    def _get_prompt_text(self) -> str:
        """Get prompt text based on state."""
        if self._state == InputState.MULTI_LINE:
            return "... "
        return "> "
    
    async def get_input(self) -> InputResult:
        """
        Get user input.
        
        Returns:
            InputResult with parsed input
        """
        # Create session if needed
        if self._session is None:
            self._session = PromptSession(
                history=self._history,
                key_bindings=self._key_bindings,
                style=self._create_prompt_style(),
                editing_mode=EditingMode.EMACS,
                multiline=False,
                enable_suspend=True,
                mouse_support=True,
            )
        
        self._state = InputState.IDLE
        
        try:
            # Get input
            text = await self._session.prompt_async(
                self._get_prompt_text(),
            )
            
            self._last_input = text
            return self._registry.parse(text)
            
        except KeyboardInterrupt:
            raise
        except EOFError:
            return InputResult(text="/exit", is_command=True, command="exit")
    
    def get_last_input(self) -> Optional[str]:
        """Get last input text."""
        return self._last_input
    
    def clear_history(self) -> None:
        """Clear input history."""
        if isinstance(self._history, InMemoryHistory):
            self._history = InMemoryHistory()
    
    def set_state(self, state: InputState) -> None:
        """Set input state."""
        self._state = state


# =============================================================================
# Multi-line Editor
# =============================================================================


class MultiLineEditor:
    """
    A multi-line text editor for longer inputs.
    
    This provides a full-screen editing experience for
    composing longer messages or code.
    """
    
    def __init__(self) -> None:
        """Initialize editor."""
        self._buffer = Buffer()
        self._text_area = TextArea(
            buffer=self._buffer,
            multiline=True,
            wrap_lines=True,
        )
        
        # Create layout
        self._layout = Layout(
            HSplit([
                Window(
                    content=BufferControl(buffer=self._buffer),
                    height=10,  # 10 lines visible
                ),
            ])
        )
    
    def get_text(self) -> str:
        """Get editor content."""
        return self._buffer.text
    
    def set_text(self, text: str) -> None:
        """Set editor content."""
        self._buffer.text = text
    
    def clear(self) -> None:
        """Clear editor content."""
        self._buffer.text = ""


# =============================================================================
# Convenience Functions
# =============================================================================


def create_input_handler(
    history_path: Optional[Path] = None,
) -> InputHandler:
    """
    Create an input handler with default settings.
    
    Args:
        history_path: Optional path for history file
        
    Returns:
        Configured InputHandler
    """
    if history_path is None:
        # Default history location
        config_dir = Path.home() / ".config" / "nim-cli"
        config_dir.mkdir(parents=True, exist_ok=True)
        history_path = config_dir / "history"
    
    return InputHandler(history_file=history_path)
