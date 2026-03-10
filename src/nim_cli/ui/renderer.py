"""
Output rendering for NIM CLI.

This module provides:
- Rich-based output rendering
- Markdown rendering
- Code syntax highlighting
- Streaming display
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme as RichTheme

from nim_cli.core.streaming import StreamChunk
from nim_cli.ui.theme import Theme, get_theme
from nim_cli.ui.animations import TypingAnimation


# =============================================================================
# Code Detection and Extraction
# =============================================================================


@dataclass
class CodeBlock:
    """Represents a code block in the response."""
    
    language: str
    code: str
    start: int
    end: int


class CodeExtractor:
    """Extracts and tracks code blocks from text."""
    
    # Pattern for code blocks
    CODE_BLOCK_PATTERN = r'```(\w*)\n(.*?)```'
    
    def __init__(self) -> None:
        """Initialize extractor."""
        self._blocks: list[CodeBlock] = []
        self._in_block = False
        self._current_block_start = 0
        self._current_language = ""
        self._current_code = ""
    
    def update(self, text: str) -> list[CodeBlock]:
        """
        Update with new text and return complete blocks.
        
        Args:
            text: New text to process
            
        Returns:
            List of newly completed code blocks
        """
        completed_blocks = []
        
        i = 0
        while i < len(text):
            if self._in_block:
                # Look for closing ```
                close_idx = text.find("```", i)
                if close_idx != -1:
                    self._current_code += text[i:close_idx]
                    block = CodeBlock(
                        language=self._current_language,
                        code=self._current_code.strip(),
                        start=self._current_block_start,
                        end=close_idx + 3,
                    )
                    self._blocks.append(block)
                    completed_blocks.append(block)
                    
                    self._in_block = False
                    self._current_code = ""
                    i = close_idx + 3
                else:
                    self._current_code += text[i:]
                    break
            else:
                # Look for opening ```
                open_idx = text.find("```", i)
                if open_idx != -1:
                    self._in_block = True
                    self._current_block_start = open_idx
                    
                    # Extract language
                    end_of_line = text.find("\n", open_idx)
                    if end_of_line != -1:
                        self._current_language = text[open_idx + 3:end_of_line].strip()
                        i = end_of_line + 1
                    else:
                        self._current_language = ""
                        i = open_idx + 3
                else:
                    break
        
        return completed_blocks
    
    def reset(self) -> None:
        """Reset extractor state."""
        self._blocks.clear()
        self._in_block = False
        self._current_block_start = 0
        self._current_language = ""
        self._current_code = ""
    
    @property
    def blocks(self) -> list[CodeBlock]:
        """Get all extracted blocks."""
        return self._blocks.copy()


# =============================================================================
# Markdown Renderer
# =============================================================================


class MarkdownRenderer:
    """
    Renders markdown with syntax highlighting for code blocks.
    
    This class provides enhanced markdown rendering with:
    - Custom styling
    - Code syntax highlighting
    - Inline code support
    """
    
    def __init__(
        self,
        code_theme: str = "monokai",
        inline_code_bg: bool = True,
    ) -> None:
        """
        Initialize markdown renderer.
        
        Args:
            code_theme: Pygments theme for code blocks
            inline_code_bg: Whether to highlight inline code background
        """
        self._code_theme = code_theme
        self._inline_code_bg = inline_code_bg
        self._code_extractor = CodeExtractor()
    
    def render(self, text: str) -> RenderableType:
        """
        Render markdown text.
        
        Args:
            text: Markdown text to render
            
        Returns:
            Rich renderable
        """
        theme = get_theme()
        
        # Check for code blocks
        self._code_extractor.reset()
        blocks = self._code_extractor.update(text)
        
        if blocks:
            # Render with custom code handling
            return self._render_with_code_blocks(text, blocks)
        
        # Simple markdown
        return Markdown(
            text,
            code_theme=self._code_theme,
            inline_code_theme=self._code_theme,
        )
    
    def _render_with_code_blocks(
        self,
        text: str,
        blocks: list[CodeBlock],
    ) -> RenderableType:
        """Render text with code blocks."""
        theme = get_theme()
        renderables = []
        
        last_end = 0
        for block in blocks:
            # Add text before code block
            if block.start > last_end:
                before_text = text[last_end:block.start]
                renderables.append(Markdown(before_text))
            
            # Add code block with syntax highlighting
            try:
                syntax = Syntax(
                    block.code,
                    block.language or "text",
                    theme=self._code_theme,
                    line_numbers=True,
                    word_wrap=True,
                    background_color=str(theme.colors.code_background),
                )
                renderables.append(syntax)
            except Exception:
                # Fallback for unknown languages
                renderables.append(Panel(
                    block.code,
                    title=block.language or "code",
                    border_style=Style(color=theme.colors.code_border.name),
                ))
            
            last_end = block.end
        
        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:]
            renderables.append(Markdown(remaining))
        
        return Group(*renderables)
    
    def render_streaming(
        self,
        text: str,
        is_complete: bool = False,
    ) -> RenderableType:
        """
        Render markdown during streaming.
        
        Handles incomplete markdown gracefully.
        
        Args:
            text: Current text
            is_complete: Whether streaming is complete
            
        Returns:
            Rich renderable
        """
        if is_complete:
            return self.render(text)
        
        # During streaming, check for incomplete code blocks
        if text.count("```") % 2 == 1:
            # Incomplete code block - add placeholder close
            return self.render(text + "\n```")
        
        return self.render(text)


# =============================================================================
# Message Renderer
# =============================================================================


@dataclass
class MessageStats:
    """Statistics for a message."""
    
    tokens: int = 0
    duration_ms: float = 0
    speed_tps: float = 0


class MessageRenderer:
    """
    Renders chat messages with proper styling.
    
    This class handles:
    - User messages
    - Assistant messages
    - System messages
    - Error messages
    """
    
    def __init__(self, console: Optional[Console] = None) -> None:
        """
        Initialize message renderer.
        
        Args:
            console: Rich console to use
        """
        self._console = console or Console()
        self._theme = get_theme()
        self._markdown = MarkdownRenderer()
    
    def render_user(
        self,
        content: str,
        stats: Optional[MessageStats] = None,
    ) -> RenderableType:
        """Render a user message."""
        # Create header
        header = Text("You", style=self._theme.user_style)
        
        # Create content panel
        panel = Panel(
            content,
            title=header,
            border_style=Style(color=self._theme.colors.user_message.name),
            padding=(1, 2),
        )
        
        return panel
    
    def render_assistant(
        self,
        content: str,
        stats: Optional[MessageStats] = None,
        is_streaming: bool = False,
    ) -> RenderableType:
        """Render an assistant message."""
        # Create header
        header_parts = [Text("Assistant", style=self._theme.assistant_style)]
        
        # Add stats if available
        if stats:
            stats_text = Text(
                f"  • {stats.tokens} tokens • {stats.duration_ms:.0f}ms",
                style=self._theme.text_muted,
            )
            header_parts.append(stats_text)
        
        header = Text.assemble(*header_parts)
        
        # Render content as markdown
        rendered_content = self._markdown.render_streaming(content, not is_streaming)
        
        # Create panel
        panel = Panel(
            rendered_content,
            title=header,
            border_style=Style(color=self._theme.colors.assistant_message.name),
            padding=(1, 2),
        )
        
        return panel
    
    def render_system(
        self,
        content: str,
        message_type: str = "info",
    ) -> RenderableType:
        """
        Render a system message.
        
        Args:
            content: Message content
            message_type: Type of message (info, warning, error, success)
        """
        # Get style based on type
        styles = {
            "info": (self._theme.info, self._theme.colors.info.name),
            "warning": (self._theme.warning, self._theme.colors.warning.name),
            "error": (self._theme.error, self._theme.colors.error.name),
            "success": (self._theme.success, self._theme.colors.success.name),
        }
        
        text_style, border_color = styles.get(message_type, styles["info"])
        
        # Create content
        content_text = Text(content, style=text_style)
        
        panel = Panel(
            content_text,
            border_style=Style(color=border_color),
            padding=(0, 1),
        )
        
        return panel
    
    def render_error(
        self,
        error: Exception,
        hint: Optional[str] = None,
    ) -> RenderableType:
        """Render an error message."""
        theme = get_theme()
        
        # Create error content
        content = Text()
        content.append(f"Error: {type(error).__name__}\n", style=theme.error)
        content.append(str(error), style=theme.text)
        
        if hint:
            content.append(f"\n\nHint: {hint}", style=theme.text_muted)
        
        panel = Panel(
            content,
            title="Error",
            border_style=Style(color=theme.colors.error.name),
            padding=(1, 2),
        )
        
        return panel


# =============================================================================
# Streaming Display
# =============================================================================


class StreamingDisplay:
    """
    Manages live display during streaming responses.
    
    This class provides a smooth streaming display with:
    - Typing animation effect
    - Markdown rendering
    - Progress tracking
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        typing_speed: int = 0,  # ms per char, 0 = instant
    ) -> None:
        """
        Initialize streaming display.
        
        Args:
            console: Rich console to use
            typing_speed: Typing animation speed (0 = instant)
        """
        self._console = console or Console()
        self._typing_speed = typing_speed
        self._theme = get_theme()
        self._markdown = MarkdownRenderer()
        self._renderer = MessageRenderer(console)
        
        # State
        self._content = ""
        self._start_time: Optional[float] = None
        self._token_count = 0
        self._typing = TypingAnimation(speed=typing_speed)
        self._live: Optional[Live] = None
    
    def start(self, model_name: Optional[str] = None) -> None:
        """
        Start streaming display.
        
        Args:
            model_name: Optional model name for header
        """
        self._content = ""
        self._start_time = time.monotonic()
        self._token_count = 0
        self._typing.reset()
        
        # Create live display
        self._live = Live(
            self._render_current(),
            console=self._console,
            refresh_per_second=30,
            transient=False,
        )
        self._live.start()
    
    def update(self, chunk: StreamChunk) -> None:
        """
        Update display with new chunk.
        
        Args:
            chunk: New stream chunk
        """
        if self._live is None:
            return
        
        self._content += chunk.content
        self._token_count += chunk.token_count
        self._typing.text = self._content
        
        # Update display
        self._live.update(self._render_current(streaming=True))
    
    def finish(self, finish_reason: Optional[str] = None) -> str:
        """
        Finish streaming and return final content.
        
        Args:
            finish_reason: Reason for completion
            
        Returns:
            Complete content
        """
        if self._live is None:
            return self._content
        
        # Calculate stats
        duration_ms = 0
        if self._start_time:
            duration_ms = (time.monotonic() - self._start_time) * 1000
        
        stats = MessageStats(
            tokens=self._token_count,
            duration_ms=duration_ms,
            speed_tps=self._token_count / (duration_ms / 1000) if duration_ms > 0 else 0,
        )
        
        # Final render
        final_render = self._renderer.render_assistant(
            self._content,
            stats=stats,
            is_streaming=False,
        )
        
        self._live.update(final_render)
        self._live.stop()
        self._live = None
        
        return self._content
    
    def _render_current(self, streaming: bool = False) -> RenderableType:
        """Render current state."""
        theme = get_theme()
        
        # Create header
        header = Text("Assistant", style=theme.assistant_style)
        if streaming:
            header.append(" ●", style=Style(color=theme.colors.accent.name))
        
        # Render content
        if self._content:
            content = self._markdown.render_streaming(
                self._content,
                is_complete=not streaming,
            )
        else:
            content = Text("Thinking...", style=theme.text_muted)
        
        # Add stats if available
        if self._start_time and self._token_count > 0:
            elapsed = (time.monotonic() - self._start_time) * 1000
            stats = Text(
                f"  • {self._token_count} tokens • {elapsed:.0f}ms",
                style=theme.text_muted,
            )
            header = Text.assemble(header, stats)
        
        panel = Panel(
            content,
            title=header,
            border_style=Style(color=theme.colors.assistant_message.name),
            padding=(1, 2),
        )
        
        return panel
    
    def cancel(self) -> None:
        """Cancel streaming display."""
        if self._live:
            self._live.stop()
            self._live = None


# =============================================================================
# Console Factory
# =============================================================================


def create_console(
    theme: Optional[Theme] = None,
    width: Optional[int] = None,
) -> Console:
    """
    Create a configured Rich console.
    
    Args:
        theme: Theme to use
        width: Optional width override
        
    Returns:
        Configured Console instance
    """
    theme = theme or get_theme()
    
    # Create Rich theme from our theme (use .name to get hex string)
    rich_theme = RichTheme({
        "primary": theme.colors.primary.name,
        "secondary": theme.colors.secondary.name,
        "accent": theme.colors.accent.name,
        "success": theme.colors.success.name,
        "warning": theme.colors.warning.name,
        "error": theme.colors.error.name,
        "info": theme.colors.info.name,
        "text": theme.colors.text.name,
        "text.muted": theme.colors.text_muted.name,
        "text.highlight": theme.colors.text_highlight.name,
        "background": theme.colors.background.name,
        "surface": theme.colors.surface.name,
    })
    
    return Console(
        theme=rich_theme,
        width=width,
        highlighter=None,  # We handle highlighting ourselves
    )
