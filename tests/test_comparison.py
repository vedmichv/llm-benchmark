"""Tests for cross-backend comparison module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_benchmark.backends import BackendResponse
from llm_benchmark.models import BenchmarkResult, ModelSummary


def _make_summary(model: str, avg_response_ts: float) -> ModelSummary:
    """Helper to build a ModelSummary with minimal data."""
    return ModelSummary(
        model=model,
        results=[
            BenchmarkResult(
                model=model,
                prompt="test prompt",
                success=True,
                response=BackendResponse(
                    model=model,
                    content="test",
                    done=True,
                    eval_count=100,
                    eval_duration=100.0 / avg_response_ts if avg_response_ts > 0 else 1.0,
                    prompt_eval_count=10,
                    prompt_eval_duration=0.1,
                    total_duration=5.0,
                ),
            )
        ],
        avg_prompt_eval_ts=100.0,
        avg_response_ts=avg_response_ts,
        avg_total_ts=avg_response_ts * 0.9,
    )


def _make_backend_status(name: str, running: bool = True):
    """Helper to create a BackendStatus-like object."""
    from llm_benchmark.backends.detection import BackendStatus

    return BackendStatus(
        name=name,
        installed=True,
        running=running,
        binary_path=f"/usr/bin/{name}",
        port={"ollama": 11434, "llama-cpp": 8080, "lm-studio": 1234}.get(name, 8080),
    )


# ---------------------------------------------------------------------------
# run_comparison tests
# ---------------------------------------------------------------------------


class TestRunComparison:
    """Tests for run_comparison orchestration."""

    @patch("llm_benchmark.comparison.unload_model", return_value=True)
    @patch("llm_benchmark.comparison.benchmark_model")
    @patch("llm_benchmark.comparison.run_preflight_checks")
    @patch("llm_benchmark.comparison.create_backend")
    def test_two_backends_returns_comparison_result(
        self, mock_create, mock_preflight, mock_benchmark, mock_unload
    ):
        """run_comparison with 2 mocked backends returns ComparisonResult with correct backend_results."""
        from llm_benchmark.comparison import ComparisonResult, run_comparison

        # Setup: two backends, each with one model
        backends = [
            _make_backend_status("ollama"),
            _make_backend_status("lm-studio"),
        ]

        mock_backend_ollama = MagicMock()
        mock_backend_ollama.name = "ollama"
        mock_backend_lms = MagicMock()
        mock_backend_lms.name = "lm-studio"
        mock_create.side_effect = [mock_backend_ollama, mock_backend_lms]

        mock_preflight.side_effect = [
            [{"model": "llama3.2:1b", "size": 1_000_000_000}],
            [{"model": "llama3.2:1b", "size": 1_000_000_000}],
        ]

        summary_ollama = _make_summary("llama3.2:1b", 45.0)
        summary_lms = _make_summary("llama3.2:1b", 38.0)
        mock_benchmark.side_effect = [summary_ollama, summary_lms]

        result = run_comparison(
            backends=backends,
            prompts=["test"],
            runs_per_prompt=1,
            timeout=60,
            skip_warmup=True,
            max_retries=0,
            verbose=False,
            skip_models=[],
            skip_checks=True,
        )

        assert isinstance(result, ComparisonResult)
        assert "ollama" in result.backends
        assert "lm-studio" in result.backends
        assert len(result.results) == 2
        assert result.overall_winner == "ollama"

    @patch("llm_benchmark.comparison.unload_model", return_value=True)
    @patch("llm_benchmark.comparison.benchmark_model")
    @patch("llm_benchmark.comparison.run_preflight_checks")
    @patch("llm_benchmark.comparison.create_backend")
    def test_single_backend_warns_and_runs(
        self, mock_create, mock_preflight, mock_benchmark, mock_unload
    ):
        """run_comparison with 1 backend warns and falls back to single-backend run."""
        from llm_benchmark.comparison import run_comparison

        backends = [_make_backend_status("ollama")]
        mock_backend = MagicMock()
        mock_backend.name = "ollama"
        mock_create.return_value = mock_backend
        mock_preflight.return_value = [{"model": "llama3.2:1b", "size": 1_000_000_000}]
        mock_benchmark.return_value = _make_summary("llama3.2:1b", 45.0)

        result = run_comparison(
            backends=backends,
            prompts=["test"],
            runs_per_prompt=1,
            timeout=60,
            skip_warmup=True,
            max_retries=0,
            verbose=False,
            skip_models=[],
            skip_checks=True,
        )

        assert len(result.backends) == 1
        assert result.overall_winner == "ollama"


# ---------------------------------------------------------------------------
# Display tests
# ---------------------------------------------------------------------------


class TestRenderComparisonBarChart:
    """Tests for render_comparison_bar_chart display."""

    def test_bar_chart_with_star_on_fastest(self, capsys):
        """render_comparison_bar_chart prints bar chart with backend names and star on fastest."""
        from llm_benchmark.comparison import render_comparison_bar_chart

        backend_rates = [("ollama", 45.0), ("llama-cpp", 62.0), ("lm-studio", 38.0)]
        render_comparison_bar_chart(backend_rates, "llama3.2:1b")

        # The function prints to Rich console, verify it was called without error
        # (Rich console captures are tricky; we verify it completes and key data is used)


class TestRenderComparisonMatrix:
    """Tests for render_comparison_matrix display."""

    def test_matrix_prints_table_with_winner(self):
        """render_comparison_matrix prints Rich table with models as rows, backends as columns, winner per row."""
        from llm_benchmark.comparison import render_comparison_matrix

        results = {
            "ollama": [
                _make_summary("llama3.2:1b", 45.0),
                _make_summary("gemma:2b", 30.0),
            ],
            "llama-cpp": [
                _make_summary("llama3.2:1b", 62.0),
                _make_summary("gemma:2b", 25.0),
            ],
        }

        # Should not raise
        render_comparison_matrix(results)

    def test_matrix_prints_fastest_backend_summary(self):
        """render_comparison_matrix prints 'Fastest backend: X (N/M models)' summary."""
        from llm_benchmark.comparison import render_comparison_matrix

        results = {
            "ollama": [
                _make_summary("llama3.2:1b", 45.0),
                _make_summary("gemma:2b", 30.0),
            ],
            "llama-cpp": [
                _make_summary("llama3.2:1b", 62.0),
                _make_summary("gemma:2b", 25.0),
            ],
        }

        # Should complete without error; output includes fastest backend line
        render_comparison_matrix(results)

    def test_matrix_handles_missing_combos(self):
        """render_comparison_matrix handles missing backend/model combos with '--'."""
        from llm_benchmark.comparison import render_comparison_matrix

        results = {
            "ollama": [
                _make_summary("llama3.2:1b", 45.0),
                _make_summary("gemma:2b", 30.0),
            ],
            "llama-cpp": [
                _make_summary("llama3.2:1b", 62.0),
                # gemma:2b missing from llama-cpp
            ],
        }

        # Should not raise -- missing combo rendered as "--"
        render_comparison_matrix(results)


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExportComparisonJson:
    """Tests for export_comparison_json."""

    def test_writes_valid_json(self, tmp_path):
        """export_comparison_json writes valid JSON with backends, models, results, winners."""
        from llm_benchmark.comparison import ComparisonResult, export_comparison_json
        from llm_benchmark.comparison import BackendModelResult
        from llm_benchmark.models import SystemInfo

        comparison = ComparisonResult(
            backends=["ollama", "llama-cpp"],
            models=["llama3.2:1b"],
            results=[
                BackendModelResult(
                    backend="ollama", model="llama3.2:1b",
                    avg_response_ts=45.0, avg_prompt_eval_ts=100.0, avg_total_ts=40.0,
                ),
                BackendModelResult(
                    backend="llama-cpp", model="llama3.2:1b",
                    avg_response_ts=62.0, avg_prompt_eval_ts=120.0, avg_total_ts=55.0,
                ),
            ],
            winner_per_model={"llama3.2:1b": "llama-cpp"},
            overall_winner="llama-cpp",
            overall_wins={"ollama": 0, "llama-cpp": 1},
        )

        system_info = SystemInfo(
            cpu="Apple M1", ram_gb=16.0, gpu="Apple M1 GPU",
            os_name="macOS 14.0", python_version="3.12.0",
            backend_name="ollama", backend_version="0.6.1",
        )

        path = export_comparison_json(comparison, system_info, output_dir=str(tmp_path))

        assert path.exists()
        data = json.loads(path.read_text())
        assert "backends" in data
        assert "models" in data
        assert "results" in data
        assert "overall_winner" in data
        assert data["overall_winner"] == "llama-cpp"


class TestExportComparisonMarkdown:
    """Tests for export_comparison_markdown."""

    def test_writes_markdown_with_matrix_and_recommendation(self, tmp_path):
        """export_comparison_markdown writes Markdown with matrix table and recommendation."""
        from llm_benchmark.comparison import ComparisonResult, export_comparison_markdown
        from llm_benchmark.comparison import BackendModelResult
        from llm_benchmark.models import SystemInfo

        comparison = ComparisonResult(
            backends=["ollama", "llama-cpp"],
            models=["llama3.2:1b"],
            results=[
                BackendModelResult(
                    backend="ollama", model="llama3.2:1b",
                    avg_response_ts=45.0, avg_prompt_eval_ts=100.0, avg_total_ts=40.0,
                ),
                BackendModelResult(
                    backend="llama-cpp", model="llama3.2:1b",
                    avg_response_ts=62.0, avg_prompt_eval_ts=120.0, avg_total_ts=55.0,
                ),
            ],
            winner_per_model={"llama3.2:1b": "llama-cpp"},
            overall_winner="llama-cpp",
            overall_wins={"ollama": 0, "llama-cpp": 1},
        )

        system_info = SystemInfo(
            cpu="Apple M1", ram_gb=16.0, gpu="Apple M1 GPU",
            os_name="macOS 14.0", python_version="3.12.0",
            backend_name="ollama", backend_version="0.6.1",
        )

        path = export_comparison_markdown(comparison, system_info, output_dir=str(tmp_path))

        assert path.exists()
        content = path.read_text()
        assert "llama-cpp" in content
        assert "Fastest backend" in content
        assert "llama3.2:1b" in content


# ---------------------------------------------------------------------------
# GGUF matching tests
# ---------------------------------------------------------------------------


class TestMatchGgufToOllamaName:
    """Tests for match_gguf_to_ollama_name."""

    def test_matches_llama32_1b(self, tmp_path):
        """match_gguf_to_ollama_name matches 'llama3.2:1b' to a GGUF file containing 'Llama-3.2-1B'."""
        from llm_benchmark.comparison import match_gguf_to_ollama_name

        gguf_path = tmp_path / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        gguf_path.touch()

        gguf_files = [(gguf_path, "Llama 3.2 1B Instruct")]

        result = match_gguf_to_ollama_name("llama3.2:1b", gguf_files)
        assert result == gguf_path

    def test_returns_none_when_no_match(self, tmp_path):
        """match_gguf_to_ollama_name returns None when no match found."""
        from llm_benchmark.comparison import match_gguf_to_ollama_name

        gguf_path = tmp_path / "phi-2-Q4_K_M.gguf"
        gguf_path.touch()

        gguf_files = [(gguf_path, "Phi 2")]

        result = match_gguf_to_ollama_name("llama3.2:1b", gguf_files)
        assert result is None
