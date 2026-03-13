"""Tests for CLI argument parsing and subcommand dispatch."""

from __future__ import annotations

from unittest.mock import patch

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
