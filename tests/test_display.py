"""Tests for the bar chart display module."""

from __future__ import annotations

from unittest.mock import patch

from llm_benchmark.display import BAR_FULL, render_bar_chart, render_text_bar_chart


class TestRenderBarChart:
    """Tests for render_bar_chart (Rich console output)."""

    @patch("llm_benchmark.display.get_console")
    def test_render_bar_chart_output(self, mock_get_console):
        """Bar chart prints model names, bars, rates, and recommendation."""
        mock_console = mock_get_console.return_value
        rankings = [("model_a", 40.0), ("model_b", 20.0)]
        render_bar_chart(rankings)

        printed = " ".join(
            str(call.args[0]) if call.args else ""
            for call in mock_console.print.call_args_list
        )

        assert "model_a" in printed
        assert "model_b" in printed
        assert BAR_FULL in printed
        assert "40.0" in printed
        assert "20.0" in printed
        assert "Best for your setup" in printed

    @patch("llm_benchmark.display.get_console")
    def test_render_bar_chart_empty(self, mock_get_console):
        """Empty rankings list produces no output and no crash."""
        mock_console = mock_get_console.return_value
        render_bar_chart([])
        mock_console.print.assert_not_called()


class TestRenderTextBarChart:
    """Tests for render_text_bar_chart (plain text for Markdown)."""

    def test_returns_string(self):
        """Return value is a plain string."""
        rankings = [("model_a", 40.0), ("model_b", 20.0)]
        result = render_text_bar_chart(rankings)
        assert isinstance(result, str)

    def test_contains_model_names_and_rates(self):
        """Output contains model names, rates, and recommendation."""
        rankings = [("model_a", 40.0), ("model_b", 20.0)]
        result = render_text_bar_chart(rankings)

        assert "model_a" in result
        assert "model_b" in result
        assert "40.0" in result
        assert "20.0" in result
        assert "Best for your setup" in result

    def test_no_rich_markup(self):
        """Plain text output has no Rich markup tags."""
        rankings = [("model_a", 40.0)]
        result = render_text_bar_chart(rankings)
        assert "[bold]" not in result
        assert "[/bold]" not in result

    def test_sorted_fastest_first(self):
        """Fastest model appears first in output lines."""
        # Input already sorted fastest-first
        rankings = [("fast_model", 100.0), ("slow_model", 10.0)]
        result = render_text_bar_chart(rankings)

        fast_pos = result.index("fast_model")
        slow_pos = result.index("slow_model")
        assert fast_pos < slow_pos

    def test_bar_proportions(self):
        """Model with half the rate gets roughly half the bar width."""
        rankings = [("top", 100.0), ("half", 50.0)]
        result = render_text_bar_chart(rankings)
        lines = result.strip().split("\n")

        # Find lines with bar characters
        top_bar = [ln for ln in lines if "top" in ln][0]
        half_bar = [ln for ln in lines if "half" in ln][0]

        top_count = top_bar.count(BAR_FULL)
        half_count = half_bar.count(BAR_FULL)

        # top should have 30 full blocks, half should have ~15
        assert top_count == 30
        assert 14 <= half_count <= 16

    def test_empty_returns_empty_string(self):
        """Empty rankings returns empty string."""
        assert render_text_bar_chart([]) == ""
