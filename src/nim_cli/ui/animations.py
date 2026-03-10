"""
Animation and visual effects for NIM CLI.

This module provides:
- Custom spinners
- Typing animations
- Progress indicators
- Visual effects
"""

from dataclasses import dataclass
from typing import Iterator, Optional

from rich.console import Console, RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.style import Style
from rich.text import Text

from nim_cli.ui.theme import Theme, get_theme


# =============================================================================
# Custom Spinners
# =============================================================================


# Custom spinner frames
NIM_SPINNERS = {
    # Classic dots with gradient effect
    "dots_pulse": {
        "interval": 80,
        "frames": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    },
    # Arc spinner
    "arc": {
        "interval": 100,
        "frames": ["◜", "◠", "◝", "◞", "◡", "◟"],
    },
    # Circle spinner
    "circle": {
        "interval": 120,
        "frames": ["◡", "⊙", "◠", "⊙"],
    },
    # Square spinner
    "square": {
        "interval": 150,
        "frames": ["◰", "◳", "◲", "◱"],
    },
    # Arrow spinner
    "arrow": {
        "interval": 100,
        "frames": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    },
    # Star spinner
    "star": {
        "interval": 100,
        "frames": ["✶", "✸", "✹", "✺", "✹", "✷"],
    },
    # Moon spinner
    "moon": {
        "interval": 150,
        "frames": ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],
    },
    # Clock spinner
    "clock": {
        "interval": 100,
        "frames": ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"],
    },
    # AI/Brain spinner
    "brain": {
        "interval": 120,
        "frames": ["🧠", "💭", "💡", "✨"],
    },
    # Loading bar
    "loading_bar": {
        "interval": 80,
        "frames": [
            "[    ]",
            "[=   ]",
            "[==  ]",
            "[=== ]",
            "[====]",
            "[ ===]",
            "[  ==]",
            "[   =]",
            "[    ]",
        ],
    },
    # Wave effect
    "wave": {
        "interval": 100,
        "frames": ["⎺", "⎻", "⎼", "⎽", "⎼", "⎻"],
    },
}


@dataclass
class SpinnerStyle:
    """Style configuration for a spinner."""
    
    name: str
    interval: int
    frames: list[str]
    style: Optional[Style] = None
    
    @classmethod
    def from_dict(cls, name: str, data: dict) -> "SpinnerStyle":
        """Create from dictionary."""
        return cls(
            name=name,
            interval=data["interval"],
            frames=data["frames"],
        )


def get_spinner(name: str = "dots_pulse") -> SpinnerStyle:
    """
    Get a spinner by name.
    
    Args:
        name: Spinner name
        
    Returns:
        SpinnerStyle instance
        
    Raises:
        KeyError: If spinner name not found
    """
    if name not in NIM_SPINNERS:
        raise KeyError(f"Unknown spinner: {name}. Available: {list(NIM_SPINNERS.keys())}")
    
    return SpinnerStyle.from_dict(name, NIM_SPINNERS[name])


# =============================================================================
# Typing Animation
# =============================================================================


class TypingAnimation:
    """
    Creates a typing animation effect for streaming text.
    
    This class provides smooth character-by-character display
    that simulates typing, with configurable speed.
    """
    
    def __init__(
        self,
        text: str = "",
        speed: int = 0,  # ms per character, 0 = instant
        cursor: str = "▌",
        cursor_style: Optional[Style] = None,
    ) -> None:
        """
        Initialize typing animation.
        
        Args:
            text: Text to animate
            speed: Milliseconds per character (0 = instant)
            cursor: Cursor character
            cursor_style: Style for cursor
        """
        self._text = text
        self._speed = speed
        self._cursor = cursor
        self._cursor_style = cursor_style or Style(bold=True)
        self._position = 0
    
    @property
    def text(self) -> str:
        """Get current text."""
        return self._text
    
    @text.setter
    def text(self, value: str) -> None:
        """Set text."""
        self._text = value
    
    @property
    def position(self) -> int:
        """Get current position."""
        return self._position
    
    def reset(self) -> None:
        """Reset to beginning."""
        self._position = 0
    
    def complete(self) -> None:
        """Skip to end."""
        self._position = len(self._text)
    
    def render(self, include_cursor: bool = True) -> Text:
        """
        Render current state.
        
        Args:
            include_cursor: Whether to show cursor
            
        Returns:
            Rich Text object
        """
        theme = get_theme()
        
        # Get visible portion
        visible = self._text[:self._position]
        
        # Create text with cursor
        result = Text(visible, style=theme.text)
        
        if include_cursor and self._position < len(self._text):
            result.append(self._cursor, style=self._cursor_style)
        
        return result
    
    def advance(self, count: int = 1) -> bool:
        """
        Advance position.
        
        Args:
            count: Number of characters to advance
            
        Returns:
            True if advanced, False if at end
        """
        if self._position >= len(self._text):
            return False
        
        self._position = min(self._position + count, len(self._text))
        return True
    
    def get_speed_delay(self) -> float:
        """Get delay per character in seconds."""
        return self._speed / 1000.0
    
    def __iter__(self) -> Iterator[Text]:
        """Iterate through animation frames."""
        self.reset()
        
        while self._position < len(self._text):
            yield self.render()
            self.advance()
        
        # Final frame without cursor
        yield self.render(include_cursor=False)


# =============================================================================
# Progress Indicators
# =============================================================================


class NIMProgress:
    """
    Custom progress indicator for NIM CLI.
    
    Provides a styled progress bar with spinner and time tracking.
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        spinner_name: str = "dots_pulse",
    ) -> None:
        """
        Initialize progress indicator.
        
        Args:
            console: Rich console to use
            spinner_name: Name of spinner to use
        """
        self._console = console or Console()
        self._spinner_name = spinner_name
        
        theme = get_theme()
        
        # Create progress bar
        self._progress = Progress(
            SpinnerColumn(spinner_name=spinner_name, style=Style(color=theme.spinner_primary.name)),
            TextColumn("[progress.description]{task.description}", style=theme.text),
            BarColumn(
                complete_style=Style(color=theme.progress_complete.name),
                finished_style=Style(color=theme.success.name),
                pulse_style=Style(color=theme.progress_incomplete.name),
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%", style=theme.text_muted),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
        )
        
        self._tasks: dict[str, TaskID] = {}
    
    def start(self) -> None:
        """Start progress display."""
        self._progress.start()
    
    def stop(self) -> None:
        """Stop progress display."""
        self._progress.stop()
    
    def add_task(
        self,
        name: str,
        description: str,
        total: Optional[float] = None,
    ) -> TaskID:
        """
        Add a new task.
        
        Args:
            name: Task identifier
            description: Task description
            total: Total steps (None for indeterminate)
            
        Returns:
            Task ID
        """
        task_id = self._progress.add_task(description, total=total)
        self._tasks[name] = task_id
        return task_id
    
    def update(
        self,
        name: str,
        *,
        advance: float = 0,
        completed: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Update task progress.
        
        Args:
            name: Task identifier
            advance: Amount to advance
            completed: Set completed amount
            description: New description
        """
        if name not in self._tasks:
            return
        
        self._progress.update(
            self._tasks[name],
            advance=advance,
            completed=completed,
            description=description,
        )
    
    def complete_task(self, name: str, description: Optional[str] = None) -> None:
        """Mark a task as complete."""
        if name not in self._tasks:
            return
        
        task_id = self._tasks[name]
        total = self._progress.tasks[task_id].total
        
        if total:
            self._progress.update(task_id, completed=total, description=description)
    
    def remove_task(self, name: str) -> None:
        """Remove a task."""
        if name not in self._tasks:
            return
        
        self._progress.remove_task(self._tasks[name])
        del self._tasks[name]
    
    def __enter__(self) -> "NIMProgress":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


# =============================================================================
# Visual Effects
# =============================================================================


class VisualEffects:
    """
    Collection of visual effects for the CLI.
    """
    
    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize with console."""
        self._console = console or Console()
    
    def flash(self, text: str, style: Optional[Style] = None) -> None:
        """
        Flash text briefly (attention grabber).
        
        Args:
            text: Text to flash
            style: Style to apply
        """
        theme = get_theme()
        style = style or Style(color=theme.accent.name, bold=True)
        
        self._console.print(text, style=style)
    
    def pulse(self, renderable: RenderableType, times: int = 3) -> Iterator[RenderableType]:
        """
        Create pulsing effect.
        
        Args:
            renderable: Content to pulse
            times: Number of pulses
            
        Yields:
            Renderable with alternating visibility
        """
        for i in range(times * 2):
            if i % 2 == 0:
                yield renderable
            else:
                yield ""
    
    def fade_in(self, text: str, steps: int = 5) -> Iterator[Text]:
        """
        Create fade-in effect.
        
        Args:
            text: Text to fade in
            steps: Number of fade steps
            
        Yields:
            Text with increasing brightness
        """
        theme = get_theme()
        
        for i in range(steps + 1):
            # Calculate alpha
            alpha = i / steps
            
            # Create faded style
            faded_style = Style(
                color=theme.colors.text,
            )
            
            yield Text(text, style=faded_style)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_spinner(name: str = "dots_pulse") -> Spinner:
    """
    Create a Rich Spinner with NIM CLI styling.
    
    Args:
        name: Spinner name
        
    Returns:
        Rich Spinner instance
    """
    theme = get_theme()
    spinner_style = get_spinner(name)
    
    return Spinner(
        name=spinner_style.name,
        style=Style(color=theme.spinner_primary.name),
        speed=spinner_style.interval / 1000.0,
    )


def get_spinner_names() -> list[str]:
    """Get list of available spinner names."""
    return list(NIM_SPINNERS.keys())
