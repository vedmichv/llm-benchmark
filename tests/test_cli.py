"""Tests for CLI argument parsing and subcommand dispatch."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRunSubcommand:
    """Tests for the 'run' subcommand."""

    def test_run_subcommand_parsing(self):
        """Verify 'run' subcommand parses without error."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"
        assert args.verbose is False
        assert args.skip_checks is False
        assert args.skip_models == []
        assert args.prompt_set == "medium"
        assert args.prompts is None
        assert args.runs_per_prompt == 2
        assert args.timeout == 200
        assert args.skip_warmup is False
        assert args.max_retries == 3

    def test_run_with_all_flags(self):
        """Verify all run flags parse correctly."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "run",
            "--verbose",
            "--skip-checks",
            "--skip-models", "model1", "model2",
            "--prompt-set", "large",
            "--runs-per-prompt", "3",
            "--timeout", "600",
            "--skip-warmup",
            "--max-retries", "5",
        ])
        assert args.verbose is True
        assert args.skip_checks is True
        assert args.skip_models == ["model1", "model2"]
        assert args.prompt_set == "large"
        assert args.runs_per_prompt == 3
        assert args.timeout == 600
        assert args.skip_warmup is True
        assert args.max_retries == 5


class TestCompareSubcommand:
    """Tests for the 'compare' subcommand."""

    def test_compare_subcommand_parsing(self):
        """Verify 'compare' subcommand parses with file arguments."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["compare", "file1.json", "file2.json"])
        assert args.command == "compare"
        assert args.files == ["file1.json", "file2.json"]

    def test_compare_with_labels(self):
        """Verify --labels parse correctly."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "compare", "f1.json", "f2.json", "--labels", "Before", "After"
        ])
        assert args.labels == ["Before", "After"]


class TestInfoSubcommand:
    """Tests for the 'info' subcommand."""

    def test_info_subcommand_parsing(self):
        """Verify 'info' subcommand parses without additional args."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"


class TestHelpAndDebug:
    """Tests for --help output and --debug flag."""

    def test_help_shows_subcommands(self, capsys):
        """--help output contains run, compare, info subcommands."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "run" in captured.out
        assert "compare" in captured.out
        assert "info" in captured.out

    def test_debug_flag(self):
        """--debug flag sets debug mode."""
        from llm_benchmark.cli import main
        from llm_benchmark.config import is_debug

        # Mock _handle_info to avoid actual system calls
        with patch("llm_benchmark.cli._handle_info", return_value=0):
            main(["--debug", "info"])
            assert is_debug() is True

        # Reset debug state
        from llm_benchmark.config import set_debug
        set_debug(False)


class TestSkipWarmup:
    """Tests for --skip-warmup CLI flag."""

    def test_skip_warmup_flag_parsed(self):
        """--skip-warmup flag parsed as True."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--skip-warmup"])
        assert args.skip_warmup is True

    def test_skip_warmup_default_false(self):
        """Default is False when --skip-warmup not provided."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.skip_warmup is False


class TestMaxRetries:
    """Tests for --max-retries CLI flag."""

    def test_max_retries_parsed(self):
        """--max-retries 5 parsed as integer 5."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--max-retries", "5"])
        assert args.max_retries == 5

    def test_max_retries_zero(self):
        """--max-retries 0 parsed as integer 0."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--max-retries", "0"])
        assert args.max_retries == 0

    def test_max_retries_default(self):
        """Default is 3 when --max-retries not provided."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.max_retries == 3


class TestBackendFlag:
    """Tests for --backend, --port, and --model-path CLI flags."""

    def test_backend_default_is_ollama(self):
        """Default backend is 'ollama' when not specified."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.backend == "ollama"

    def test_backend_llama_cpp(self):
        """--backend llama-cpp parses correctly."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--backend", "llama-cpp"])
        assert args.backend == "llama-cpp"

    def test_backend_lm_studio(self):
        """--backend lm-studio parses correctly."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--backend", "lm-studio"])
        assert args.backend == "lm-studio"

    def test_backend_invalid_rejected(self):
        """Invalid backend value rejected by argparse."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--backend", "invalid"])

    def test_port_flag_parsed_as_int(self):
        """--port flag parsed as integer."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--port", "5555"])
        assert args.port == 5555

    def test_port_default_is_none(self):
        """Default port is None when not specified."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.port is None

    def test_model_path_flag_parsed(self):
        """--model-path flag parsed as string."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--model-path", "/path/to/model.gguf"])
        assert args.model_path == "/path/to/model.gguf"

    def test_model_path_default_is_none(self):
        """Default model_path is None when not specified."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.model_path is None

    def test_backend_and_port_combined(self):
        """--backend lm-studio --port 5555 parses together."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--backend", "lm-studio", "--port", "5555"])
        assert args.backend == "lm-studio"
        assert args.port == 5555

    def test_info_backend_flag(self):
        """Info subcommand accepts --backend flag."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["info", "--backend", "llama-cpp"])
        assert args.backend == "llama-cpp"

    def test_info_backend_default(self):
        """Info subcommand defaults to ollama backend."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["info"])
        assert args.backend == "ollama"


class TestBackendAll:
    """Tests for --backend all (cross-backend comparison mode)."""

    def test_backend_all_accepted_by_parser(self):
        """--backend all is accepted without error."""
        from llm_benchmark.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--backend", "all"])
        assert args.backend == "all"

    def test_backend_all_triggers_comparison_branch(self):
        """--backend all calls run_comparison and render_comparison_bar_chart for single model."""
        from llm_benchmark.cli import _handle_run

        args = argparse.Namespace(
            command="run",
            backend="all",
            prompt_set="medium",
            prompts=None,
            runs_per_prompt=2,
            timeout=200,
            skip_warmup=False,
            max_retries=3,
            verbose=False,
            skip_models=[],
            skip_checks=False,
        )

        mock_status = MagicMock()
        mock_status.running = True
        mock_status.name = "ollama"

        mock_result = MagicMock()
        mock_result.backend = "ollama"
        mock_result.avg_response_ts = 42.0

        mock_comparison = MagicMock()
        mock_comparison.backends = ["ollama"]
        mock_comparison.models = ["llama3.2:1b"]
        mock_comparison.results = [mock_result]

        mock_system_info = MagicMock()

        with (
            patch(
                "llm_benchmark.backends.detection.detect_backends",
                return_value=[mock_status],
            ) as mock_detect,
            patch(
                "llm_benchmark.comparison.run_comparison",
                return_value=mock_comparison,
            ) as mock_run_comp,
            patch(
                "llm_benchmark.comparison.export_comparison_json",
                return_value=Path("results/comparison.json"),
            ),
            patch(
                "llm_benchmark.comparison.export_comparison_markdown",
                return_value=Path("results/comparison.md"),
            ),
            patch("llm_benchmark.system.get_system_info", return_value=mock_system_info),
            patch("llm_benchmark.backends.create_backend", return_value=MagicMock()),
            patch("llm_benchmark.cli.get_console"),
            patch(
                "llm_benchmark.comparison.render_comparison_bar_chart",
            ) as mock_bar_chart,
            patch(
                "llm_benchmark.comparison.render_comparison_matrix",
            ) as mock_matrix,
        ):
            result = _handle_run(args)

        assert result == 0
        mock_detect.assert_called_once()
        mock_run_comp.assert_called_once()
        mock_bar_chart.assert_called_once_with(
            [("ollama", 42.0)], "llama3.2:1b"
        )
        mock_matrix.assert_not_called()

    def test_backend_all_multi_model_calls_matrix(self):
        """--backend all with 2+ models calls render_comparison_matrix, not bar chart."""
        from llm_benchmark.cli import _handle_run

        args = argparse.Namespace(
            command="run",
            backend="all",
            prompt_set="medium",
            prompts=None,
            runs_per_prompt=2,
            timeout=200,
            skip_warmup=False,
            max_retries=3,
            verbose=False,
            skip_models=[],
            skip_checks=False,
        )

        mock_status = MagicMock()
        mock_status.running = True
        mock_status.name = "ollama"

        mock_result_a = MagicMock()
        mock_result_a.backend = "ollama"
        mock_result_a.model = "model-a"
        mock_result_a.avg_response_ts = 50.0

        mock_result_b = MagicMock()
        mock_result_b.backend = "ollama"
        mock_result_b.model = "model-b"
        mock_result_b.avg_response_ts = 35.0

        mock_comparison = MagicMock()
        mock_comparison.backends = ["ollama"]
        mock_comparison.models = ["model-a", "model-b"]
        mock_comparison.results = [mock_result_a, mock_result_b]

        mock_system_info = MagicMock()

        with (
            patch(
                "llm_benchmark.backends.detection.detect_backends",
                return_value=[mock_status],
            ),
            patch(
                "llm_benchmark.comparison.run_comparison",
                return_value=mock_comparison,
            ),
            patch(
                "llm_benchmark.comparison.export_comparison_json",
                return_value=Path("results/comparison.json"),
            ),
            patch(
                "llm_benchmark.comparison.export_comparison_markdown",
                return_value=Path("results/comparison.md"),
            ),
            patch("llm_benchmark.system.get_system_info", return_value=mock_system_info),
            patch("llm_benchmark.backends.create_backend", return_value=MagicMock()),
            patch("llm_benchmark.cli.get_console"),
            patch(
                "llm_benchmark.comparison.render_comparison_bar_chart",
            ) as mock_bar_chart,
            patch(
                "llm_benchmark.comparison.render_comparison_matrix",
            ) as mock_matrix,
        ):
            result = _handle_run(args)

        assert result == 0
        mock_matrix.assert_called_once()
        mock_bar_chart.assert_not_called()
        # Verify the dict passed to render_comparison_matrix
        call_args = mock_matrix.call_args[0][0]
        assert "ollama" in call_args
        assert len(call_args["ollama"]) == 2

    def test_backend_all_no_running_backends_returns_error(self):
        """--backend all with 0 running backends returns exit code 1."""
        from llm_benchmark.cli import _handle_run

        args = argparse.Namespace(
            command="run",
            backend="all",
            prompt_set="medium",
            prompts=None,
            runs_per_prompt=2,
            timeout=200,
            skip_warmup=False,
            max_retries=3,
            verbose=False,
            skip_models=[],
            skip_checks=False,
        )

        mock_status = MagicMock()
        mock_status.running = False

        with (
            patch(
                "llm_benchmark.backends.detection.detect_backends",
                return_value=[mock_status],
            ),
            patch("llm_benchmark.cli.get_console"),
        ):
            result = _handle_run(args)

        assert result == 1


class TestCreateBackendFactory:
    """Tests for create_backend() factory function."""

    def test_create_backend_ollama(self):
        """create_backend('ollama') returns OllamaBackend."""
        from llm_benchmark.backends import create_backend
        from llm_benchmark.backends.ollama import OllamaBackend

        backend = create_backend("ollama")
        assert isinstance(backend, OllamaBackend)

    def test_create_backend_llama_cpp(self):
        """create_backend('llama-cpp') returns LlamaCppBackend."""
        from llm_benchmark.backends import create_backend
        from llm_benchmark.backends.llamacpp import LlamaCppBackend

        backend = create_backend("llama-cpp")
        assert isinstance(backend, LlamaCppBackend)

    def test_create_backend_lm_studio(self):
        """create_backend('lm-studio') returns LMStudioBackend."""
        from llm_benchmark.backends import create_backend
        from llm_benchmark.backends.lmstudio import LMStudioBackend

        backend = create_backend("lm-studio")
        assert isinstance(backend, LMStudioBackend)

    def test_create_backend_invalid_raises(self):
        """create_backend('invalid') raises ValueError."""
        from llm_benchmark.backends import create_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("invalid")

    def test_create_backend_with_port(self):
        """create_backend('llama-cpp', port=9090) uses custom port."""
        from llm_benchmark.backends import create_backend

        backend = create_backend("llama-cpp", port=9090)
        assert "9090" in backend._base_url

    def test_create_backend_with_host_and_port(self):
        """create_backend('lm-studio', host='0.0.0.0', port=5555) uses custom host and port."""
        from llm_benchmark.backends import create_backend

        backend = create_backend("lm-studio", host="0.0.0.0", port=5555)
        assert "0.0.0.0" in backend._base_url
        assert "5555" in backend._base_url

    def test_create_backend_default_is_ollama(self):
        """create_backend() with no args returns OllamaBackend."""
        from llm_benchmark.backends import create_backend
        from llm_benchmark.backends.ollama import OllamaBackend

        backend = create_backend()
        assert isinstance(backend, OllamaBackend)
