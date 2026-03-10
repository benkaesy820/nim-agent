"""
Entry point for NIM CLI.

Usage:
    python -m nim_cli
    nim-cli
"""

import sys


def main() -> int:
    """Main entry point for the CLI."""
    # Import here to avoid circular imports and speed up startup
    from nim_cli.cli import Application
    
    try:
        app = Application()
        return app.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
