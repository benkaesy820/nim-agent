"""
UI module for NIM CLI.

This module provides all user interface components:
- Theme system
- Rendering
- Animations
- Input handling
"""

from nim_cli.ui.theme import (
    Theme,
    ThemeType,
    Colors,
    DARK_COLORS,
    LIGHT_COLORS,
    MONOKAI_COLORS,
    DRACULA_COLORS,
    NORD_COLORS,
    ThemeManager,
    get_theme,
    set_theme,
)
from nim_cli.ui.renderer import (
    MarkdownRenderer,
    MessageRenderer,
    MessageStats,
    StreamingDisplay,
    CodeBlock,
    CodeExtractor,
    create_console,
)
from nim_cli.ui.animations import (
    NIM_SPINNERS,
    SpinnerStyle,
    TypingAnimation,
    NIMProgress,
    VisualEffects,
    get_spinner,
    get_spinner_names,
    create_spinner,
)
from nim_cli.ui.input import (
    InputHandler,
    InputState,
    InputResult,
    Command,
    CommandRegistry,
    MultiLineEditor,
    create_input_handler,
)

__all__ = [
    # Theme
    "Theme",
    "ThemeType",
    "Colors",
    "DARK_COLORS",
    "LIGHT_COLORS",
    "MONOKAI_COLORS",
    "DRACULA_COLORS",
    "NORD_COLORS",
    "ThemeManager",
    "get_theme",
    "set_theme",
    
    # Renderer
    "MarkdownRenderer",
    "MessageRenderer",
    "MessageStats",
    "StreamingDisplay",
    "CodeBlock",
    "CodeExtractor",
    "create_console",
    
    # Animations
    "NIM_SPINNERS",
    "SpinnerStyle",
    "TypingAnimation",
    "NIMProgress",
    "VisualEffects",
    "get_spinner",
    "get_spinner_names",
    "create_spinner",
    
    # Input
    "InputHandler",
    "InputState",
    "InputResult",
    "Command",
    "CommandRegistry",
    "MultiLineEditor",
    "create_input_handler",
]
