"""Tests for llm_benchmark.exporters module -- cache indicators and output."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_benchmark.models import (
    BenchmarkResult,
    ModelSummary,
    OllamaResponse,
)


def _make_result(
    prompt_cached: bool = False,
    eval_count: int = 120,
    eval_duration: int = 4_000_000_000,
    prompt_eval_count: int = 15,
    prompt_eval_duration: int = 200_000_000,
    success: bool = True,
    error: str | None = None,
) -> BenchmarkResult:
    """Helper to build a BenchmarkResult."""
    if not success:
        return BenchmarkResult(
            model="test-model",
            prompt="test prompt for benchmark",
            success=False,
            error=error or "some error",
        )
    resp = OllamaResponse(
        model="test-model",
        created_at=datetime.now(timezone.utc),
        message={"role": "assistant", "content": "test response"},
        done=True,
        total_duration=prompt_eval_duration + eval_duration,
        load_duration=0,
        prompt_eval_count=prompt_eval_count,
        prompt_eval_duration=prompt_eval_duration,
        eval_count=eval_count,
        eval_duration=eval_duration,
        prompt_cached=prompt_cached,
    )
    return BenchmarkResult(
        model="test-model",
        prompt="test prompt for benchmark",
        success=True,
        response=resp,
        prompt_cached=prompt_cached,
    )


def _make_summary(results: list[BenchmarkResult] | None = None) -> ModelSummary:
    """Helper to build a ModelSummary with mixed cached/non-cached results."""
    if results is None:
        results = [
            _make_result(prompt_cached=False),
            _make_result(prompt_cached=True, prompt_eval_count=0, prompt_eval_duration=0),
        ]
    return ModelSummary(
        model="test-model",
        results=results,
        avg_prompt_eval_ts=75.0,
        avg_response_ts=30.0,
        avg_total_ts=32.14,
    )


class TestCsvCacheColumn:
    """Test that CSV export includes a Cached column."""

    def test_csv_has_cached_header(self, tmp_path):
        """CSV output includes 'Cached' column header."""
        from llm_benchmark.exporters import export_csv

        summary = _make_summary()
        filepath = export_csv([summary], output_dir=tmp_path)

        with open(filepath) as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert "Cached" in headers

    def test_csv_cached_run_shows_yes(self, tmp_path):
        """Cached run rows show 'Yes' in Cached column."""
        from llm_benchmark.exporters import export_csv

        cached_result = _make_result(prompt_cached=True, prompt_eval_count=0, prompt_eval_duration=0)
        summary = _make_summary([cached_result])
        filepath = export_csv([summary], output_dir=tmp_path)

        with open(filepath) as f:
            reader = csv.reader(f)
            headers = next(reader)
            cached_idx = headers.index("Cached")
            row = next(reader)
        assert row[cached_idx] == "Yes"

    def test_csv_non_cached_run_shows_no(self, tmp_path):
        """Non-cached run rows show 'No' in Cached column."""
        from llm_benchmark.exporters import export_csv

        non_cached_result = _make_result(prompt_cached=False)
        summary = _make_summary([non_cached_result])
        filepath = export_csv([summary], output_dir=tmp_path)

        with open(filepath) as f:
            reader = csv.reader(f)
            headers = next(reader)
            cached_idx = headers.index("Cached")
            row = next(reader)
        assert row[cached_idx] == "No"

    def test_csv_failed_run_empty_cached(self, tmp_path):
        """Failed run rows show empty string in Cached column."""
        from llm_benchmark.exporters import export_csv

        failed_result = _make_result(success=False, error="timeout")
        summary = _make_summary([failed_result])
        filepath = export_csv([summary], output_dir=tmp_path)

        with open(filepath) as f:
            reader = csv.reader(f)
            headers = next(reader)
            cached_idx = headers.index("Cached")
            row = next(reader)
        assert row[cached_idx] == ""


class TestMarkdownCacheIndicator:
    """Test that Markdown export includes [cached] indicator."""

    def test_markdown_cached_run_shows_indicator(self, tmp_path):
        """Markdown detailed results show '[cached]' for cached runs."""
        from llm_benchmark.exporters import export_markdown

        cached_result = _make_result(prompt_cached=True, prompt_eval_count=0, prompt_eval_duration=0)
        summary = _make_summary([cached_result])
        filepath = export_markdown([summary], output_dir=tmp_path)

        content = filepath.read_text()
        assert "[cached]" in content

    def test_markdown_non_cached_no_indicator(self, tmp_path):
        """Non-cached runs do not show '[cached]'."""
        from llm_benchmark.exporters import export_markdown

        non_cached_result = _make_result(prompt_cached=False)
        summary = _make_summary([non_cached_result])
        filepath = export_markdown([summary], output_dir=tmp_path)

        content = filepath.read_text()
        assert "[cached]" not in content


class TestResultsGitignore:
    """Test that results/.gitignore is auto-created."""

    def test_gitignore_created_for_results_dir(self, tmp_path):
        """When output_dir ends with 'results', .gitignore is auto-created."""
        from llm_benchmark.exporters import export_csv

        results_dir = tmp_path / "results"
        summary = _make_summary([_make_result()])
        export_csv([summary], output_dir=results_dir)

        gitignore = results_dir / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text()
        assert "*.json" in content
        assert "*.csv" in content
        assert "*.md" in content
        assert "!.gitignore" in content

    def test_gitignore_not_created_for_other_dirs(self, tmp_path):
        """Non-results directories do not get .gitignore auto-created."""
        from llm_benchmark.exporters import export_csv

        other_dir = tmp_path / "output"
        summary = _make_summary([_make_result()])
        export_csv([summary], output_dir=other_dir)

        gitignore = other_dir / ".gitignore"
        assert not gitignore.exists()


class TestExporterOutputDir:
    """Test that export functions create files under specified output directory."""

    def test_export_csv_creates_in_dir(self, tmp_path):
        """export_csv creates file under specified output directory."""
        from llm_benchmark.exporters import export_csv

        summary = _make_summary([_make_result()])
        filepath = export_csv([summary], output_dir=tmp_path)
        assert filepath.parent == tmp_path
        assert filepath.exists()

    def test_export_markdown_creates_in_dir(self, tmp_path):
        """export_markdown creates file under specified output directory."""
        from llm_benchmark.exporters import export_markdown

        summary = _make_summary([_make_result()])
        filepath = export_markdown([summary], output_dir=tmp_path)
        assert filepath.parent == tmp_path
        assert filepath.exists()

    def test_export_json_creates_in_dir(self, tmp_path):
        """export_json creates file under specified output directory."""
        from llm_benchmark.exporters import export_json

        summary = _make_summary([_make_result()])
        filepath = export_json([summary], output_dir=tmp_path)
        assert filepath.parent == tmp_path
        assert filepath.exists()
