"""Tests for the interactive menu module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_model(name: str, size: int) -> dict:
    """Create a model dict with 'model' and 'size' keys."""
    return {"model": name, "size": size}


@pytest.fixture()
def models():
    """Return a list of 3 model dicts with varying sizes."""
    return [
        _make_model("big-model:7b", 7_000_000_000),
        _make_model("small-model:1b", 1_000_000_000),
        _make_model("mid-model:3b", 3_000_000_000),
    ]


@pytest.fixture()
def mock_menu_backend():
    """Return a mock backend for menu tests."""
    backend = MagicMock()
    backend.name = "ollama"
    backend.version = "0.6.1"
    return backend


class TestQuickTestMode:
    """Quick-test mode selects smallest model and skips everything else."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda backend, m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_quick_test_mode(self, _mock_summary, _mock_console, _mock_rec, models, mock_menu_backend):
        """Mode '1' returns quick-test Namespace."""
        from llm_benchmark.menu import run_interactive_menu

        with patch("builtins.input", return_value="1"):
            ns = run_interactive_menu(mock_menu_backend, models)

        assert isinstance(ns, argparse.Namespace)
        assert ns.runs_per_prompt == 1
        assert ns.skip_warmup is True
        # Smallest model should NOT be in skip_models
        assert "small-model:1b" not in ns.skip_models
        # Other models should be skipped
        assert "big-model:7b" in ns.skip_models
        assert "mid-model:3b" in ns.skip_models

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda backend, m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_smallest_model_selection(self, _mock_summary, _mock_console, _mock_rec, mock_menu_backend):
        """Quick test picks the model with the smallest size."""
        from llm_benchmark.menu import run_interactive_menu

        models = [
            _make_model("large:70b", 70_000_000_000),
            _make_model("tiny:0.5b", 500_000_000),
            _make_model("medium:7b", 7_000_000_000),
        ]
        with patch("builtins.input", return_value="1"):
            ns = run_interactive_menu(mock_menu_backend, models)

        # tiny model should be the only one NOT skipped
        assert "tiny:0.5b" not in ns.skip_models
        assert "large:70b" in ns.skip_models
        assert "medium:7b" in ns.skip_models


class TestStandardMode:
    """Standard benchmark mode."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda backend, m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_standard_mode(self, _mock_summary, _mock_console, _mock_rec, models, mock_menu_backend):
        """Mode '2' returns standard Namespace."""
        from llm_benchmark.menu import run_interactive_menu

        with patch("builtins.input", return_value="2"):
            ns = run_interactive_menu(mock_menu_backend, models)

        assert ns.prompt_set == "medium"
        assert ns.runs_per_prompt == 2
        assert ns.skip_warmup is False


class TestFullMode:
    """Full benchmark mode."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda backend, m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_full_mode(self, _mock_summary, _mock_console, _mock_rec, models, mock_menu_backend):
        """Mode '3' returns full Namespace."""
        from llm_benchmark.menu import run_interactive_menu

        with patch("builtins.input", return_value="3"):
            ns = run_interactive_menu(mock_menu_backend, models)

        assert ns.prompt_set == "large"
        assert ns.runs_per_prompt == 3
        assert ns.skip_warmup is False


class TestCustomMode:
    """Custom mode with multiple inputs."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda backend, m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_custom_mode(self, _mock_summary, _mock_console, _mock_rec, models, mock_menu_backend):
        """Mode '4' prompts for prompt set, runs, and model skip."""
        from llm_benchmark.menu import run_interactive_menu

        # mode=4, prompt_set=2 (medium), runs=3, skip models=""
        inputs = iter(["4", "2", "3", ""])
        with patch("builtins.input", side_effect=inputs):
            ns = run_interactive_menu(mock_menu_backend, models)

        assert ns.prompt_set == "medium"
        assert ns.runs_per_prompt == 3
        assert ns.skip_models == []


class TestInvalidInput:
    """Invalid input handling and re-prompting."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda backend, m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_invalid_input_reprompts(self, _mock_summary, _mock_console, _mock_rec, models, mock_menu_backend):
        """Invalid choices cause re-prompting until valid input."""
        from llm_benchmark.menu import run_interactive_menu

        # "x" and "abc" are invalid, "1" is valid
        inputs = iter(["x", "abc", "1"])
        with patch("builtins.input", side_effect=inputs) as mock_input:
            ns = run_interactive_menu(mock_menu_backend, models)

        assert mock_input.call_count == 3
        assert ns.runs_per_prompt == 1  # quick-test mode


class TestBuildNamespaceBackendField:
    """_build_namespace includes backend and model_path fields."""

    def test_default_backend_field(self):
        """Namespace includes backend='ollama' by default."""
        from llm_benchmark.menu import _build_namespace

        ns = _build_namespace()
        assert ns.backend == "ollama"
        assert ns.model_path is None
        assert ns.port is None

    def test_custom_backend_field(self):
        """Namespace includes the specified backend name."""
        from llm_benchmark.menu import _build_namespace

        ns = _build_namespace(backend="llama-cpp", model_path="/tmp/model.gguf")
        assert ns.backend == "llama-cpp"
        assert ns.model_path == "/tmp/model.gguf"

    def test_backend_propagated_through_modes(self):
        """All mode functions pass backend through to _build_namespace."""
        from llm_benchmark.menu import _mode_full, _mode_standard

        ns = _mode_standard(backend="lm-studio")
        assert ns.backend == "lm-studio"

        ns = _mode_full(backend="llama-cpp", model_path="/tmp/m.gguf")
        assert ns.backend == "llama-cpp"
        assert ns.model_path == "/tmp/m.gguf"


class TestMenuBackendPropagation:
    """run_interactive_menu passes backend name through to namespace."""

    @patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda backend, m: m)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.system.format_system_summary", return_value="sys info")
    def test_backend_in_namespace(self, _mock_summary, _mock_console, _mock_rec, models):
        """Namespace includes backend name from the backend instance."""
        from llm_benchmark.menu import run_interactive_menu

        backend = MagicMock()
        backend.name = "lm-studio"

        with patch("builtins.input", return_value="2"):
            ns = run_interactive_menu(backend, models)

        assert ns.backend == "lm-studio"


class TestSelectBackendInteractive:
    """Tests for select_backend_interactive()."""

    def _make_status(self, name, installed, running, port=8080):
        from llm_benchmark.backends.detection import BackendStatus
        return BackendStatus(
            name=name,
            installed=installed,
            running=running,
            binary_path=f"/usr/bin/{name}" if installed else None,
            port=port,
        )

    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.backends.detection.detect_backends")
    def test_single_backend_auto_selects(self, mock_detect, _mock_console):
        """When only one backend is available, it auto-selects."""
        from llm_benchmark.menu import select_backend_interactive

        mock_detect.return_value = [
            self._make_status("ollama", True, True, 11434),
            self._make_status("llama-cpp", False, False, 8080),
            self._make_status("lm-studio", False, False, 1234),
        ]

        name, port, model_path = select_backend_interactive()
        assert name == "ollama"
        assert port is None
        assert model_path is None

    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.backends.detection.detect_backends")
    def test_multiple_backends_prompts_choice(self, mock_detect, _mock_console):
        """When multiple backends available, prompts user to choose."""
        from llm_benchmark.menu import select_backend_interactive

        mock_detect.return_value = [
            self._make_status("ollama", True, True, 11434),
            self._make_status("llama-cpp", False, False, 8080),
            self._make_status("lm-studio", True, True, 1234),
        ]

        # User picks option 3 (lm-studio)
        with patch("builtins.input", return_value="3"):
            name, port, model_path = select_backend_interactive()

        assert name == "lm-studio"

    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.backends.detection.detect_backends")
    def test_no_backends_defaults_ollama(self, mock_detect, _mock_console):
        """When no backends detected, defaults to ollama."""
        from llm_benchmark.menu import select_backend_interactive

        mock_detect.return_value = [
            self._make_status("ollama", False, False, 11434),
            self._make_status("llama-cpp", False, False, 8080),
            self._make_status("lm-studio", False, False, 1234),
        ]

        name, port, model_path = select_backend_interactive()
        assert name == "ollama"

    @patch("llm_benchmark.menu._select_gguf_model", return_value="/tmp/model.gguf")
    @patch("llm_benchmark.menu._prompt_yn", return_value=False)
    @patch("llm_benchmark.menu.get_console")
    @patch("llm_benchmark.backends.detection.detect_backends")
    def test_llamacpp_not_running_asks_gguf(self, mock_detect, _mock_console, _mock_yn, _mock_gguf):
        """llama-cpp installed but not running triggers GGUF selection."""
        from llm_benchmark.menu import select_backend_interactive

        mock_detect.return_value = [
            self._make_status("ollama", False, False, 11434),
            self._make_status("llama-cpp", True, False, 8080),
            self._make_status("lm-studio", False, False, 1234),
        ]

        name, port, model_path = select_backend_interactive()
        assert name == "llama-cpp"
        assert model_path == "/tmp/model.gguf"


class TestGGUFScanning:
    """Tests for GGUF file scanning utilities."""

    def test_scan_gguf_files_empty_dir(self, tmp_path):
        """Empty directory returns empty list."""
        from llm_benchmark.menu import scan_gguf_files

        result = scan_gguf_files(tmp_path)
        assert result == []

    def test_scan_gguf_files_nonexistent_dir(self):
        """Non-existent directory returns empty list."""
        from llm_benchmark.menu import scan_gguf_files

        result = scan_gguf_files(Path("/nonexistent/path"))
        assert result == []

    def test_scan_gguf_files_finds_files(self, tmp_path):
        """Finds .gguf files and uses filename as display name."""
        from llm_benchmark.menu import scan_gguf_files

        # Create fake gguf files
        (tmp_path / "model-a.gguf").write_bytes(b"fake")
        (tmp_path / "model-b.gguf").write_bytes(b"fake")
        (tmp_path / "not-a-model.txt").write_bytes(b"fake")

        result = scan_gguf_files(tmp_path)
        assert len(result) == 2
        paths = [str(p) for p, _ in result]
        assert any("model-a.gguf" in p for p in paths)
        assert any("model-b.gguf" in p for p in paths)

    def test_scan_gguf_files_nested(self, tmp_path):
        """Finds .gguf files in nested directories."""
        from llm_benchmark.menu import scan_gguf_files

        nested = tmp_path / "hub" / "models"
        nested.mkdir(parents=True)
        (nested / "deep-model.gguf").write_bytes(b"fake")

        result = scan_gguf_files(tmp_path)
        assert len(result) == 1
        assert "deep model" in result[0][1]  # display name from filename

    def test_extract_gguf_model_name_invalid_file(self, tmp_path):
        """Non-GGUF files return None."""
        from llm_benchmark.menu import extract_gguf_model_name

        fake = tmp_path / "fake.gguf"
        fake.write_bytes(b"not a gguf file")
        assert extract_gguf_model_name(fake) is None

    def test_extract_gguf_model_name_missing_file(self):
        """Missing file returns None."""
        from llm_benchmark.menu import extract_gguf_model_name

        assert extract_gguf_model_name(Path("/nonexistent/model.gguf")) is None

    def test_clean_gguf_filename(self):
        """Filename cleaning strips .gguf and replaces separators."""
        from llm_benchmark.menu import _clean_gguf_filename

        assert _clean_gguf_filename(Path("my-model-Q4_K_M.gguf")) == "my model Q4 K M"


class TestFailureSummary:
    """Tests for the failure summary display in cli._print_failure_summary."""

    @patch("llm_benchmark.cli.get_console")
    def test_failure_summary_display(self, mock_get_console):
        """Failure summary prints a table with model names and errors."""
        from llm_benchmark.cli import _print_failure_summary

        mock_console = MagicMock()
        mock_get_console.return_value = mock_console

        failures = [
            ("model-a:7b", "connection refused"),
            ("model-b:3b", "timeout after 120s"),
        ]
        _print_failure_summary(failures, "ollama")

        # Should have called rule and print
        mock_console.rule.assert_called_once()
        # Should print the table and count message
        assert mock_console.print.call_count >= 2

    @patch("llm_benchmark.cli.get_console")
    def test_failure_summary_with_hints(self, mock_get_console):
        """Known issues produce hints in the failure summary."""
        from llm_benchmark.runner import get_known_issue_hint

        # Verify the hint mechanism works
        hint = get_known_issue_hint("ollama", "connection refused")
        assert hint is not None
        assert "ollama serve" in hint.lower() or "running" in hint.lower()
