# NIM CLI

> World-class agentic CLI powered by NVIDIA NIM

A lightweight, robust, and beautiful terminal interface for AI-powered conversations and coding assistance.

## Features

- **Free API**: Powered by NVIDIA NIM free tier
- **Beautiful UI**: Rich terminal rendering with animations
- **Streaming**: Real-time token-by-token display
- **Multi-line Input**: Support for complex prompts
- **Command System**: Built-in commands for control
- **Themes**: Multiple color themes (dark, light, monokai, dracula, nord)
- **Lightweight**: Minimal dependencies, fast startup

## Installation

```bash
# Clone and install
git clone https://github.com/nim-cli/nim-cli
cd nim-cli
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

```bash
# Set your API key
export NVIDIA_API_KEY="nvapi-xxx"

# Run
nim-cli
```

## Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/help` | `/h`, `/?` | Show available commands |
| `/exit` | `/q`, `/quit` | Exit the CLI |
| `/clear` | `/c`, `/cls` | Clear conversation |
| `/model [name]` | `/m` | Switch or show model |
| `/models` | `/ml` | List available models |
| `/config [key] [value]` | | View/modify config |
| `/theme [name]` | | Change theme |
| `/stats` | | Show session statistics |
| `/debug` | | Toggle debug mode |

## Available Models

- `meta/llama-3.1-8b-instruct` (default)
- `meta/llama-3.1-70b-instruct`
- `meta/llama-3.1-405b-instruct`
- `nvidia/llama-3.1-nemotron-70b-instruct`
- `deepseek-ai/deepseek-r1`
- And more...

## Configuration

Config file location: `~/.config/nim-cli/config.toml`

```toml
[model]
name = "meta/llama-3.1-8b-instruct"
temperature = 0.7
max_tokens = 4096

[display]
theme = "dark"
animation_speed = 1.0
```

## Key Bindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+C` | Cancel / Exit |
| `Ctrl+D` | Exit |
| `Ctrl+L` | Clear screen |
| `Tab` | Autocomplete command |
| `Up/Down` | Navigate history |

## Architecture

```
nim-cli/
├── core/           # Core functionality
│   ├── config.py   # Configuration management
│   ├── client.py   # NIM API client
│   ├── streaming.py # SSE handling
│   ├── retry.py    # Retry & circuit breaker
│   └── errors.py   # Error handling
├── ui/             # User interface
│   ├── theme.py    # Themes & colors
│   ├── renderer.py # Output rendering
│   ├── input.py    # Input handling
│   └── animations.py # Visual effects
└── cli.py          # Main application
```

## Performance

- **uvloop**: 2-4x faster event loop
- **orjson**: 3-10x faster JSON parsing
- **Connection pooling**: 60% latency reduction
- **TCP_NODELAY**: 40-60ms saved per request

## License

MIT
