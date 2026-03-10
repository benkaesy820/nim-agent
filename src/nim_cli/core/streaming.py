"""
Streaming engine for NVIDIA NIM API responses.

This module handles:
- SSE (Server-Sent Events) parsing
- Token buffering for display optimization
- Markdown state tracking
- Streaming metrics collection
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Optional,
    Union,
)

import orjson

from nim_cli.core.errors import (
    NIMContentFilterError,
    NIMModelError,
    StreamError,
    StreamInterruptedError,
    StreamParseError,
)
from nim_cli.core.metrics import get_metrics


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StreamChunk:
    """
    A chunk of streaming response data.
    
    Attributes:
        content: Text content in this chunk
        is_complete: Whether streaming has finished
        finish_reason: Reason for completion (stop, length, content_filter)
        token_count: Estimated tokens in this chunk
        latency_ms: Time since request start
        accumulated_content: All content received so far
    """
    
    content: str
    is_complete: bool = False
    finish_reason: Optional[str] = None
    token_count: int = 0
    latency_ms: float = 0.0
    accumulated_content: str = ""
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"StreamChunk(content={preview!r}, complete={self.is_complete})"


@dataclass
class StreamMetrics:
    """Metrics for a streaming response."""
    
    start_time: float = field(default_factory=time.monotonic)
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    chunks_received: int = 0
    total_tokens: int = 0
    total_content_length: int = 0
    
    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000
    
    @property
    def total_time_ms(self) -> Optional[float]:
        """Total streaming time in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Token throughput rate."""
        if self.end_time is None or self.total_tokens == 0:
            return None
        duration = self.end_time - self.start_time
        if duration <= 0:
            return None
        return self.total_tokens / duration


# =============================================================================
# Markdown State Machine
# =============================================================================


class MarkdownState(Enum):
    """States for markdown parsing."""
    TEXT = auto()
    CODE_BLOCK = auto()
    INLINE_CODE = auto()
    BOLD = auto()
    ITALIC = auto()
    HEADER = auto()


@dataclass
class MarkdownTracker:
    """
    Tracks markdown state for proper rendering.
    
    This helps determine when we have complete markdown elements
    for optimal display updates.
    """
    
    state: MarkdownState = MarkdownState.TEXT
    buffer: str = ""
    code_block_language: str = ""
    pending_delimiters: int = 0
    
    # Track open/close pairs
    in_code_block: bool = False
    in_inline_code: bool = False
    in_bold: bool = False
    in_italic: bool = False
    
    def update(self, text: str) -> None:
        """Update state based on new text."""
        self.buffer += text
        
        # Check for code blocks (highest priority)
        if "```" in text:
            self._handle_code_block(text)
            return
        
        # Check for inline code
        if "`" in text and not self.in_code_block:
            self._handle_inline_code(text)
            return
        
        # Check for bold/italic
        if "**" in text or "__" in text:
            self._handle_bold(text)
        if "*" in text or "_" in text:
            self._handle_italic(text)
    
    def _handle_code_block(self, text: str) -> None:
        """Handle code block delimiters."""
        if self.in_code_block:
            # Look for closing
            if text.rstrip().endswith("```"):
                self.in_code_block = False
                self.code_block_language = ""
        else:
            # Look for opening
            match = re.search(r"```(\w*)", text)
            if match:
                self.in_code_block = True
                self.code_block_language = match.group(1)
    
    def _handle_inline_code(self, text: str) -> None:
        """Handle inline code delimiters."""
        count = text.count("`")
        if count % 2 == 1:
            self.in_inline_code = not self.in_inline_code
    
    def _handle_bold(self, text: str) -> None:
        """Handle bold delimiters."""
        count = text.count("**") + text.count("__")
        if count % 2 == 1:
            self.in_bold = not self.in_bold
    
    def _handle_italic(self, text: str) -> None:
        """Handle italic delimiters."""
        # Avoid confusion with bold
        if "**" not in text and "__" not in text:
            count = text.count("*") + text.count("_")
            # Filter out double delimiters used for bold
            count = count // 2 if count > 2 else count
            if count % 2 == 1:
                self.in_italic = not self.in_italic
    
    def is_at_boundary(self) -> bool:
        """Check if we're at a safe rendering boundary."""
        # Safe to render if not in middle of any markdown element
        return not (
            self.in_code_block
            or self.in_inline_code
            or self.in_bold
            or self.in_italic
        )
    
    def get_pending_content(self) -> str:
        """Get content that's been buffered but not yet returned."""
        return self.buffer


# =============================================================================
# SSE Parser
# =============================================================================


class SSEParser:
    """
    Parses Server-Sent Events from NVIDIA NIM API.
    
    NVIDIA NIM uses OpenAI-compatible SSE format:
    ```
    data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hi"},...}]}
    data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" there"},...}]}
    data: [DONE]
    ```
    """
    
    # Pattern for SSE data lines
    DATA_PATTERN = re.compile(r"^data:\s*(.+)$", re.MULTILINE)
    
    def __init__(self) -> None:
        """Initialize SSE parser."""
        self._buffer = ""
    
    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        """
        Feed a chunk of bytes to the parser.
        
        Args:
            chunk: Raw bytes from the response stream
            
        Returns:
            List of parsed JSON data objects
        """
        # Decode the chunk
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            # Buffer incomplete UTF-8
            self._buffer += chunk.decode("utf-8", errors="replace")
            return []
        
        # Add to buffer
        self._buffer += text
        
        # Extract complete data lines
        results = []
        
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            
            if not line:
                continue
            
            if line == "data: [DONE]":
                # Signal end of stream
                results.append({"done": True})
                continue
            
            match = self.DATA_PATTERN.match(line)
            if match:
                data_str = match.group(1)
                try:
                    data = orjson.loads(data_str)
                    results.append(data)
                except orjson.JSONDecodeError as e:
                    # Log but don't fail - try to continue
                    raise StreamParseError(
                        f"Failed to parse SSE data: {e}",
                        raw_data=data_str,
                    )
        
        return results
    
    def reset(self) -> None:
        """Reset parser state."""
        self._buffer = ""


# =============================================================================
# Stream Processor
# =============================================================================


class StreamProcessor:
    """
    Processes streaming responses with smart buffering.
    
    This class:
    - Parses SSE events from the API
    - Buffers tokens for optimal display
    - Tracks markdown state for clean rendering
    - Collects metrics
    """
    
    # Minimum buffer size before flush (characters)
    MIN_BUFFER_SIZE = 1
    
    # Maximum time to hold buffer (seconds)
    MAX_BUFFER_TIME = 0.1
    
    # Characters that trigger immediate flush
    FLUSH_TRIGGERS = {".", "!", "?", "\n", " ", ":", ";", ","}
    
    def __init__(
        self,
        flush_callback: Optional[Callable[[StreamChunk], None]] = None,
    ) -> None:
        """
        Initialize stream processor.
        
        Args:
            flush_callback: Optional callback when buffer is flushed
        """
        self._parser = SSEParser()
        self._markdown = MarkdownTracker()
        self._metrics = StreamMetrics()
        self._flush_callback = flush_callback
        
        # Buffering state
        self._buffer = ""
        self._last_flush_time = time.monotonic()
        self._accumulated = ""
        
        # Error tracking
        self._had_error = False
        self._error_content = ""
    
    @property
    def metrics(self) -> StreamMetrics:
        """Get streaming metrics."""
        return self._metrics
    
    @property
    def accumulated_content(self) -> str:
        """Get all content accumulated so far."""
        return self._accumulated
    
    async def process(
        self,
        stream: AsyncIterator[bytes],
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a stream of bytes into StreamChunks.
        
        Args:
            stream: Async iterator of bytes from the API
            
        Yields:
            StreamChunk objects with content
        """
        try:
            async for chunk in stream:
                # Parse SSE events
                try:
                    events = self._parser.feed(chunk)
                except StreamParseError as e:
                    self._had_error = True
                    self._error_content = self._accumulated
                    raise
                
                # Process each event
                for event in events:
                    async for stream_chunk in self._process_event(event):
                        yield stream_chunk
                        
                        if self._flush_callback:
                            self._flush_callback(stream_chunk)
            
            # Flush any remaining buffer
            if self._buffer:
                yield self._create_chunk(
                    self._buffer,
                    is_complete=False,
                )
                self._buffer = ""
        
        except asyncio.CancelledError:
            # Stream was cancelled
            raise StreamInterruptedError(
                "Stream was cancelled",
                partial_content=self._accumulated,
            )
        
        finally:
            # Record end time
            self._metrics.end_time = time.monotonic()
    
    async def _process_event(
        self,
        event: dict[str, Any],
    ) -> AsyncGenerator[StreamChunk, None]:
        """Process a single SSE event."""
        
        # Check for done signal
        if event.get("done"):
            # Flush remaining buffer
            if self._buffer:
                yield self._create_chunk(
                    self._buffer,
                    is_complete=True,
                    finish_reason="stop",
                )
                self._buffer = ""
            return
        
        # Extract content from OpenAI format
        try:
            choices = event.get("choices", [])
            if not choices:
                return
            
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            finish_reason = choices[0].get("finish_reason")
            
            # Check for content filter
            if finish_reason == "content_filter":
                raise NIMContentFilterError(
                    reason="Response was filtered by content policy",
                )
            
            if content:
                # Track first token time
                if self._metrics.first_token_time is None:
                    self._metrics.first_token_time = time.monotonic()
                
                # Update metrics
                self._metrics.chunks_received += 1
                self._metrics.total_content_length += len(content)
                
                # Estimate tokens (rough: ~4 chars per token)
                estimated_tokens = max(1, len(content) // 4)
                self._metrics.total_tokens += estimated_tokens
                
                # Update markdown state
                self._markdown.update(content)
                
                # Buffer the content
                self._buffer += content
                self._accumulated += content
                
                # Check if we should flush
                should_flush = self._should_flush(content)
                
                if should_flush:
                    yield self._create_chunk(
                        self._buffer,
                        is_complete=False,
                        token_count=estimated_tokens,
                    )
                    self._buffer = ""
                    self._last_flush_time = time.monotonic()
            
            # Handle finish
            if finish_reason:
                if self._buffer:
                    yield self._create_chunk(
                        self._buffer,
                        is_complete=True,
                        finish_reason=finish_reason,
                    )
                    self._buffer = ""
        
        except KeyError as e:
            raise StreamParseError(
                f"Unexpected event structure: missing {e}",
                raw_data=str(event),
            )
    
    def _should_flush(self, new_content: str) -> bool:
        """Determine if buffer should be flushed."""
        # Always flush if at markdown boundary and have content
        if self._buffer and self._markdown.is_at_boundary():
            # Check for sentence endings or whitespace
            if new_content and new_content[-1] in self.FLUSH_TRIGGERS:
                return True
        
        # Force flush if buffer is getting large
        if len(self._buffer) >= 100:
            return True
        
        # Force flush if too much time has passed
        if time.monotonic() - self._last_flush_time > self.MAX_BUFFER_TIME:
            return True
        
        return False
    
    def _create_chunk(
        self,
        content: str,
        is_complete: bool,
        finish_reason: Optional[str] = None,
        token_count: int = 0,
    ) -> StreamChunk:
        """Create a StreamChunk with metrics."""
        current_time = time.monotonic()
        latency_ms = (current_time - self._metrics.start_time) * 1000
        
        return StreamChunk(
            content=content,
            is_complete=is_complete,
            finish_reason=finish_reason,
            token_count=token_count or max(1, len(content) // 4),
            latency_ms=latency_ms,
            accumulated_content=self._accumulated,
        )
    
    def reset(self) -> None:
        """Reset processor state for a new stream."""
        self._parser.reset()
        self._markdown = MarkdownTracker()
        self._metrics = StreamMetrics()
        self._buffer = ""
        self._last_flush_time = time.monotonic()
        self._accumulated = ""
        self._had_error = False
        self._error_content = ""


# =============================================================================
# Helper Functions
# =============================================================================


async def collect_stream(stream: AsyncIterator[bytes]) -> str:
    """
    Collect all content from a stream.
    
    Useful for testing or when you need the full response.
    
    Args:
        stream: Async iterator of bytes
        
    Returns:
        Complete content string
    """
    processor = StreamProcessor()
    content = ""
    
    async for chunk in processor.process(stream):
        content += chunk.content
    
    return content
