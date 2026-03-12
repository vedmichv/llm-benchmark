"""Shared configuration: console singleton, debug flag, and constants."""

from rich.console import Console

_console = Console()
_debug: bool = False

# Constants
DEFAULT_TIMEOUT: int = 300  # seconds
DEFAULT_PROMPT_SET: str = "medium"
DEFAULT_RUNS_PER_PROMPT: int = 2
RAM_SAFETY_MULTIPLIER: float = 1.2
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_WARMUP_PROMPT: str = "Hello"


def get_console() -> Console:
    """Return the shared Rich console instance."""
    return _console


def is_debug() -> bool:
    """Return whether debug mode is active."""
    return _debug


def set_debug(enabled: bool) -> None:
    """Enable or disable debug mode."""
    global _debug
    _debug = enabled
