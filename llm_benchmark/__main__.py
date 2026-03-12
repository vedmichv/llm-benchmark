"""Entry point for ``python -m llm_benchmark``.

Wraps cli.main() with top-level exception handling:
- KeyboardInterrupt: friendly message, exit 130
- Other exceptions: friendly error with --debug suggestion (unless debug mode)
"""

import sys

from llm_benchmark.cli import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        sys.exit(130)
    except Exception as e:
        from llm_benchmark.config import is_debug

        if is_debug():
            raise
        from llm_benchmark.config import get_console

        console = get_console()
        console.print(f"[red bold]Error:[/red bold] {e}")
        console.print("[dim]Run with --debug for full traceback[/dim]")
        sys.exit(1)
