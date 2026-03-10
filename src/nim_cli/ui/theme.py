"""
Theme definitions for NIM CLI.

This module provides:
- Color schemes and themes
- Theme management
- Rich style definitions
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rich.color import Color
from rich.style import Style
from rich.text import Text


class ThemeType(Enum):
    """Available theme types."""
    DARK = "dark"
    LIGHT = "light"
    MONOKAI = "monokai"
    DRACULA = "dracula"
    NORD = "nord"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Colors:
    """Color palette for a theme."""
    
    # Primary colors
    primary: Color
    secondary: Color
    accent: Color
    
    # Semantic colors
    success: Color
    warning: Color
    error: Color
    info: Color
    
    # Text colors
    text: Color
    text_muted: Color
    text_highlight: Color
    
    # Background colors
    background: Color
    surface: Color
    surface_highlight: Color
    
    # Message colors
    user_message: Color
    assistant_message: Color
    system_message: Color
    
    # Code colors
    code_background: Color
    code_border: Color


# =============================================================================
# Built-in Themes
# =============================================================================


DARK_COLORS = Colors(
    # Primary
    primary=Color.parse("#7C3AED"),      # Vibrant purple
    secondary=Color.parse("#3B82F6"),     # Blue
    accent=Color.parse("#10B981"),        # Emerald
    
    # Semantic
    success=Color.parse("#22C55E"),
    warning=Color.parse("#F59E0B"),
    error=Color.parse("#EF4444"),
    info=Color.parse("#3B82F6"),
    
    # Text
    text=Color.parse("#F9FAFB"),
    text_muted=Color.parse("#9CA3AF"),
    text_highlight=Color.parse("#FBBF24"),
    
    # Background
    background=Color.parse("#111827"),
    surface=Color.parse("#1F2937"),
    surface_highlight=Color.parse("#374151"),
    
    # Messages
    user_message=Color.parse("#8B5CF6"),
    assistant_message=Color.parse("#3B82F6"),
    system_message=Color.parse("#6B7280"),
    
    # Code
    code_background=Color.parse("#0D1117"),
    code_border=Color.parse("#30363D"),
)


LIGHT_COLORS = Colors(
    # Primary
    primary=Color.parse("#6366F1"),
    secondary=Color.parse("#3B82F6"),
    accent=Color.parse("#10B981"),
    
    # Semantic
    success=Color.parse("#16A34A"),
    warning=Color.parse("#D97706"),
    error=Color.parse("#DC2626"),
    info=Color.parse("#2563EB"),
    
    # Text
    text=Color.parse("#1F2937"),
    text_muted=Color.parse("#6B7280"),
    text_highlight=Color.parse("#B45309"),
    
    # Background
    background=Color.parse("#F9FAFB"),
    surface=Color.parse("#F3F4F6"),
    surface_highlight=Color.parse("#E5E7EB"),
    
    # Messages
    user_message=Color.parse("#6366F1"),
    assistant_message=Color.parse("#2563EB"),
    system_message=Color.parse("#9CA3AF"),
    
    # Code
    code_background=Color.parse("#F8FAFC"),
    code_border=Color.parse("#E2E8F0"),
)


MONOKAI_COLORS = Colors(
    # Primary
    primary=Color.parse("#A6E22E"),       # Green
    secondary=Color.parse("#66D9EF"),     # Cyan
    accent=Color.parse("#F92672"),        # Pink
    
    # Semantic
    success=Color.parse("#A6E22E"),
    warning=Color.parse("#FD971F"),
    error=Color.parse("#F92672"),
    info=Color.parse("#66D9EF"),
    
    # Text
    text=Color.parse("#F8F8F2"),
    text_muted=Color.parse("#75715E"),
    text_highlight=Color.parse("#E6DB74"),
    
    # Background
    background=Color.parse("#272822"),
    surface=Color.parse("#3E3D32"),
    surface_highlight=Color.parse("#49483E"),
    
    # Messages
    user_message=Color.parse("#A6E22E"),
    assistant_message=Color.parse("#66D9EF"),
    system_message=Color.parse("#75715E"),
    
    # Code
    code_background=Color.parse("#1E1F1C"),
    code_border=Color.parse("#3E3D32"),
)


DRACULA_COLORS = Colors(
    # Primary
    primary=Color.parse("#BD93F9"),       # Purple
    secondary=Color.parse("#8BE9FD"),     # Cyan
    accent=Color.parse("#FF79C6"),        # Pink
    
    # Semantic
    success=Color.parse("#50FA7B"),
    warning=Color.parse("#FFB86C"),
    error=Color.parse("#FF5555"),
    info=Color.parse("#8BE9FD"),
    
    # Text
    text=Color.parse("#F8F8F2"),
    text_muted=Color.parse("#6272A4"),
    text_highlight=Color.parse("#F1FA8C"),
    
    # Background
    background=Color.parse("#282A36"),
    surface=Color.parse("#44475A"),
    surface_highlight=Color.parse("#5A5F76"),
    
    # Messages
    user_message=Color.parse("#BD93F9"),
    assistant_message=Color.parse("#8BE9FD"),
    system_message=Color.parse("#6272A4"),
    
    # Code
    code_background=Color.parse("#1D1E26"),
    code_border=Color.parse("#44475A"),
)


NORD_COLORS = Colors(
    # Primary
    primary=Color.parse("#88C0D0"),       # Frost
    secondary=Color.parse("#81A1C1"),     # Frost darker
    accent=Color.parse("#A3BE8C"),        # Aurora green
    
    # Semantic
    success=Color.parse("#A3BE8C"),
    warning=Color.parse("#EBCB8B"),
    error=Color.parse("#BF616A"),
    info=Color.parse("#88C0D0"),
    
    # Text
    text=Color.parse("#ECEFF4"),          # Snow
    text_muted=Color.parse("#D8DEE9"),    # Snow darker
    text_highlight=Color.parse("#EBCB8B"),
    
    # Background
    background=Color.parse("#2E3440"),    # Polar Night
    surface=Color.parse("#3B4252"),
    surface_highlight=Color.parse("#434C5E"),
    
    # Messages
    user_message=Color.parse("#88C0D0"),
    assistant_message=Color.parse("#81A1C1"),
    system_message=Color.parse("#D8DEE9"),
    
    # Code
    code_background=Color.parse("#242933"),
    code_border=Color.parse("#3B4252"),
)


# =============================================================================
# Theme Class
# =============================================================================


class Theme:
    """
    Complete theme with styles derived from colors.
    
    This class provides Rich Style objects for all UI elements,
    making it easy to apply consistent styling throughout the app.
    """
    
    def __init__(self, colors: Colors, name: str = "custom") -> None:
        """
        Initialize theme from colors.
        
        Args:
            colors: Color palette
            name: Theme name
        """
        self.colors = colors
        self.name = name
        
        # Derive styles from colors
        self._init_styles()
    
    def _init_styles(self) -> None:
        """Initialize all style objects."""
        c = self.colors
        
        # Text styles - use .name to convert Color to hex string
        self.text = Style(color=c.text.name)
        self.text_muted = Style(color=c.text_muted.name)
        self.text_highlight = Style(color=c.text_highlight.name, bold=True)
        
        # Semantic styles
        self.success = Style(color=c.success.name)
        self.warning = Style(color=c.warning.name)
        self.error = Style(color=c.error.name, bold=True)
        self.info = Style(color=c.info.name)
        
        # Message styles
        self.user_style = Style(
            color=c.user_message.name,
            bold=True,
        )
        self.assistant_style = Style(
            color=c.assistant_message.name,
        )
        self.system_style = Style(
            color=c.system_message.name,
            italic=True,
        )
        
        # Panel styles (border colors stored separately for Panel widgets)
        self.user_panel = Style(
            color=c.text.name,
            bgcolor=c.user_message.name,
        )
        self.assistant_panel = Style(
            color=c.text.name,
            bgcolor=c.surface.name,
        )
        self.system_panel = Style(
            color=c.text_muted.name,
            bgcolor=c.surface.name,
        )
        self.error_panel = Style(
            color=c.text.name,
            bgcolor=c.surface.name,
        )
        
        # Border colors (for Panel border_style parameter) - keep as Color objects
        # These are accessed via .name when used in Panel widgets
        self.user_border = c.user_message
        self.assistant_border = c.assistant_message
        self.system_border = c.system_message
        self.error_border = c.error
        
        # Code styles
        self.code_block = Style(
            color=c.text.name,
            bgcolor=c.code_background.name,
        )
        self.inline_code = Style(
            color=c.accent.name,
            bgcolor=c.surface.name,
        )
        
        # UI element styles
        self.title = Style(
            color=c.primary.name,
            bold=True,
        )
        self.subtitle = Style(
            color=c.secondary.name,
        )
        self.prompt = Style(
            color=c.primary.name,
            bold=True,
        )
        self.cursor = Style(
            color=c.accent.name,
            blink=True,
        )
        
        # Spinner colors - keep as Color objects (accessed via .name when used)
        self.spinner_primary = c.primary
        self.spinner_secondary = c.secondary
        
        # Progress bar colors
        self.progress_complete = c.primary
        self.progress_incomplete = c.surface_highlight
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def dark(cls) -> "Theme":
        """Create dark theme."""
        return cls(DARK_COLORS, "dark")
    
    @classmethod
    def light(cls) -> "Theme":
        """Create light theme."""
        return cls(LIGHT_COLORS, "light")
    
    @classmethod
    def monokai(cls) -> "Theme":
        """Create monokai theme."""
        return cls(MONOKAI_COLORS, "monokai")
    
    @classmethod
    def dracula(cls) -> "Theme":
        """Create dracula theme."""
        return cls(DRACULA_COLORS, "dracula")
    
    @classmethod
    def nord(cls) -> "Theme":
        """Create nord theme."""
        return cls(NORD_COLORS, "nord")
    
    @classmethod
    def from_name(cls, name: str) -> "Theme":
        """
        Create theme by name.
        
        Args:
            name: Theme name
            
        Returns:
            Theme instance
            
        Raises:
            ValueError: If theme name is unknown
        """
        themes = {
            "dark": cls.dark,
            "light": cls.light,
            "monokai": cls.monokai,
            "dracula": cls.dracula,
            "nord": cls.nord,
        }
        
        if name.lower() not in themes:
            raise ValueError(f"Unknown theme: {name}. Available: {list(themes.keys())}")
        
        return themes[name.lower()]()


# =============================================================================
# Theme Manager
# =============================================================================


class ThemeManager:
    """Manages the current theme."""
    
    _instance: Optional["ThemeManager"] = None
    
    def __init__(self) -> None:
        """Initialize theme manager."""
        self._theme: Optional[Theme] = None
        self._available_themes = {
            "dark": Theme.dark,
            "light": Theme.light,
            "monokai": Theme.monokai,
            "dracula": Theme.dracula,
            "nord": Theme.nord,
        }
    
    @classmethod
    def get_instance(cls) -> "ThemeManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def theme(self) -> Theme:
        """Get current theme (lazy initialization)."""
        if self._theme is None:
            self._theme = Theme.dark()
        return self._theme
    
    def set_theme(self, name: str) -> None:
        """Set theme by name."""
        if name.lower() not in self._available_themes:
            raise ValueError(f"Unknown theme: {name}")
        self._theme = self._available_themes[name.lower()]()
    
    def list_themes(self) -> list[str]:
        """List available theme names."""
        return list(self._available_themes.keys())


def get_theme() -> Theme:
    """Get the current theme."""
    return ThemeManager.get_instance().theme


def set_theme(name: str) -> None:
    """Set the current theme by name."""
    ThemeManager.get_instance().set_theme(name)
