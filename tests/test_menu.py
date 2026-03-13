"""Tests for the interactive menu module."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


def _make_model(name: str, size: int) -> MagicMock:
    """Create a mock Ollama model object with .model and .size attrs."""
    m = MagicMock()
    m.model = name
    m.size = size
    return m


@pytest.fixture()
def models():
    """Return a list of 3 mock models with varying sizes."""
    return [
        _make_model("big-model:7b", 7_000_000_000),
        _make_model("small-model:1b", 1_000_000_000),
        _make_model("mid-model:3b", 3_000_000_000),
    ]


class TestQuickTestMode:
    """Quick-test mode selects smallest model and skips everything else."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_quick_test_mode(self, _mock_summary, _mock_console, _mock_rec, models):
        """Mode '1' returns quick-test Namespace."""
        from llm_benchmark.menu import run_interactive_menu

        with patch("builtins.input", return_value="1"):
            ns = run_interactive_menu(models)

        assert isinstance(ns, argparse.Namespace)
        assert ns.runs_per_prompt == 1
        assert ns.skip_warmup is True
        # Smallest model should NOT be in skip_models
        assert "small-model:1b" not in ns.skip_models
        # Other models should be skipped
        assert "big-model:7b" in ns.skip_models
        assert "mid-model:3b" in ns.skip_models

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_smallest_model_selection(self, _mock_summary, _mock_console, _mock_rec):
        """Quick test picks the model with the smallest .size."""
        from llm_benchmark.menu import run_interactive_menu

        models = [
            _make_model("large:70b", 70_000_000_000),
            _make_model("tiny:0.5b", 500_000_000),
            _make_model("medium:7b", 7_000_000_000),
        ]
        with patch("builtins.input", return_value="1"):
            ns = run_interactive_menu(models)

        # tiny model should be the only one NOT skipped
        assert "tiny:0.5b" not in ns.skip_models
        assert "large:70b" in ns.skip_models
        assert "medium:7b" in ns.skip_models


class TestStandardMode:
    """Standard benchmark mode."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_standard_mode(self, _mock_summary, _mock_console, _mock_rec, models):
        """Mode '2' returns standard Namespace."""
        from llm_benchmark.menu import run_interactive_menu

        with patch("builtins.input", return_value="2"):
            ns = run_interactive_menu(models)

        assert ns.prompt_set == "medium"
        assert ns.runs_per_prompt == 2
        assert ns.skip_warmup is False


class TestFullMode:
    """Full benchmark mode."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_full_mode(self, _mock_summary, _mock_console, _mock_rec, models):
        """Mode '3' returns full Namespace."""
        from llm_benchmark.menu import run_interactive_menu

        with patch("builtins.input", return_value="3"):
            ns = run_interactive_menu(models)

        assert ns.prompt_set == "large"
        assert ns.runs_per_prompt == 3
        assert ns.skip_warmup is False


class TestCustomMode:
    """Custom mode with multiple inputs."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_custom_mode(self, _mock_summary, _mock_console, _mock_rec, models):
        """Mode '4' prompts for prompt set, runs, and model skip."""
        from llm_benchmark.menu import run_interactive_menu

        # mode=4, prompt_set=2 (medium), runs=3, skip models=""
        inputs = iter(["4", "2", "3", ""])
        with patch("builtins.input", side_effect=inputs):
            ns = run_interactive_menu(models)

        assert ns.prompt_set == "medium"
        assert ns.runs_per_prompt == 3
        assert ns.skip_models == []


class TestInvalidInput:
    """Invalid input handling and re-prompting."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_invalid_input_reprompts(self, _mock_summary, _mock_console, _mock_rec, models):
        """Invalid choices cause re-prompting until valid input."""
        from llm_benchmark.menu import run_interactive_menu

        # "x" and "abc" are invalid, "1" is valid
        inputs = iter(["x", "abc", "1"])
        with patch("builtins.input", side_effect=inputs) as mock_input:
            ns = run_interactive_menu(models)

        assert mock_input.call_count == 3
        assert ns.runs_per_prompt == 1  # quick-test mode
