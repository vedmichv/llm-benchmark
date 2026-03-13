"""Bar chart rendering and recommendation display.

Provides both Rich-formatted terminal output and plain-text versions
suitable for Markdown reports.

Exports:
    render_bar_chart: Print a ranked bar chart to the terminal.
    render_text_bar_chart: Return a plain-text bar chart string.
"""

from __future__ import annotations

from llm_benchmark.config import get_console

# Unicode block characters for the bar chart.
BAR_FULL = "\u2588"
BAR_EMPTY = "\u2591"
BAR_WIDTH = 30


def render_bar_chart(
    rankings: list[tuple[str, float]],
    metric_label: str = "t/s",
) -> None:
    """Print a ranked horizontal bar chart to the Rich console.

    Parameters
    ----------
    rankings:
        List of ``(model_name, rate)`` tuples, already sorted
        fastest-first (descending).
    metric_label:
        Unit label shown after each rate value.
    """
    if not rankings:
        return

    console = get_console()
    max_rate = max(r for _, r in rankings)
    max_name_len = max(len(n) for n, _ in rankings)

    console.print()
    console.print("[bold]Model Rankings[/bold]")
    console.print()

    for name, rate in rankings:
        if max_rate > 0:
            bar_len = round(BAR_WIDTH * rate / max_rate)
        else:
            bar_len = 0
        bar = BAR_FULL * bar_len + BAR_EMPTY * (BAR_WIDTH - bar_len)
        console.print(f"  {name:<{max_name_len}}  {bar}  {rate:>6.1f} {metric_label}")

    console.print()
    best_name, best_rate = rankings[0]
    console.print(
        f"  Best for your setup: [bold]{best_name}[/bold] "
        f"({best_rate:.1f} {metric_label}) -- fastest response generation"
    )


def render_text_bar_chart(
    rankings: list[tuple[str, float]],
    metric_label: str = "t/s",
) -> str:
    """Return a plain-text bar chart string for Markdown embedding.

    Same layout as :func:`render_bar_chart` but without Rich markup or
    ANSI codes -- pure Unicode text.

    Parameters
    ----------
    rankings:
        List of ``(model_name, rate)`` tuples, already sorted
        fastest-first (descending).
    metric_label:
        Unit label shown after each rate value.

    Returns
    -------
    str
        Multi-line plain text chart.
    """
    if not rankings:
        return ""

    max_rate = max(r for _, r in rankings)
    max_name_len = max(len(n) for n, _ in rankings)

    lines: list[str] = []
    lines.append("Model Rankings")
    lines.append("")

    for name, rate in rankings:
        if max_rate > 0:
            bar_len = round(BAR_WIDTH * rate / max_rate)
        else:
            bar_len = 0
        bar = BAR_FULL * bar_len + BAR_EMPTY * (BAR_WIDTH - bar_len)
        lines.append(f"  {name:<{max_name_len}}  {bar}  {rate:>6.1f} {metric_label}")

    lines.append("")
    best_name, best_rate = rankings[0]
    lines.append(
        f"  Best for your setup: {best_name} "
        f"({best_rate:.1f} {metric_label}) -- fastest response generation"
    )

    return "\n".join(lines)
