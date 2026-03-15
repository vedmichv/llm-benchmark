"""Tests for llm_benchmark.exporters module -- cache indicators and output."""

import csv
import json

from llm_benchmark.backends import BackendResponse
from llm_benchmark.models import (
    BenchmarkResult,
    ModelSummary,
    SystemInfo,
)


def _make_result(
    prompt_cached: bool = False,
    eval_count: int = 120,
    eval_duration: float = 4.0,
    prompt_eval_count: int = 15,
    prompt_eval_duration: float = 0.2,
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
    resp = BackendResponse(
        model="test-model",
        content="test response",
        done=True,
        total_duration=prompt_eval_duration + eval_duration,
        load_duration=0.0,
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
            _make_result(prompt_cached=True, prompt_eval_count=0, prompt_eval_duration=0.0),
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

        cached_result = _make_result(prompt_cached=True, prompt_eval_count=0, prompt_eval_duration=0.0)
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

        cached_result = _make_result(prompt_cached=True, prompt_eval_count=0, prompt_eval_duration=0.0)
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


class TestMarkdownRankings:
    """Test that Markdown export includes rankings section."""

    def _make_system_info(self):
        from llm_benchmark.models import SystemInfo

        return SystemInfo(
            cpu="Apple M1",
            ram_gb=16.0,
            gpu="Apple M1 (integrated)",
            os_name="macOS 14.0",
            python_version="3.12.0",
            backend_name="ollama",
            backend_version="0.6.1",
        )

    def test_export_markdown_has_rankings(self, tmp_path):
        """Markdown report includes Rankings section with bar chart."""
        from llm_benchmark.exporters import export_markdown

        summaries = [
            ModelSummary(
                model="fast-model:1b",
                results=[_make_result()],
                avg_prompt_eval_ts=100.0,
                avg_response_ts=50.0,
                avg_total_ts=45.0,
            ),
            ModelSummary(
                model="slow-model:7b",
                results=[_make_result()],
                avg_prompt_eval_ts=40.0,
                avg_response_ts=20.0,
                avg_total_ts=18.0,
            ),
        ]
        filepath = export_markdown(
            summaries,
            system_info=self._make_system_info(),
            output_dir=tmp_path,
        )
        content = filepath.read_text()

        assert "## Rankings" in content
        assert "Best for your setup" in content
        # Bar chart chars
        assert "\u2588" in content

    def test_export_markdown_compact_header(self, tmp_path):
        """Markdown includes compact header with mode and model count."""
        from llm_benchmark.exporters import export_markdown

        summary = _make_summary([_make_result()])
        filepath = export_markdown(
            [summary],
            system_info=self._make_system_info(),
            output_dir=tmp_path,
        )
        content = filepath.read_text()

        assert "**Mode:** Standard" in content
        assert "**Models:** 1" in content

    def test_export_markdown_one_line_system_info(self, tmp_path):
        """Markdown has one-line system info, not multi-line bullets."""
        from llm_benchmark.exporters import export_markdown

        summary = _make_summary([_make_result()])
        filepath = export_markdown(
            [summary],
            system_info=self._make_system_info(),
            output_dir=tmp_path,
        )
        content = filepath.read_text()

        assert "**System:**" in content
        assert "Apple M1" in content
        assert "16.0 GB RAM" in content


def _make_system_info(backend_name: str = "ollama", backend_version: str = "0.6.1") -> SystemInfo:
    """Helper to build a SystemInfo with configurable backend."""
    return SystemInfo(
        cpu="Apple M1",
        ram_gb=16.0,
        gpu="Apple M1 (integrated)",
        os_name="macOS 14.0",
        python_version="3.12.0",
        backend_name=backend_name,
        backend_version=backend_version,
    )


class TestBackendAwareFilenames:
    """Test that export filenames include the backend name."""

    def test_json_filename_includes_backend(self, tmp_path):
        """JSON filename follows pattern: benchmark_{backend}_{timestamp}.json."""
        from llm_benchmark.exporters import export_json

        summary = _make_summary([_make_result()])
        si = _make_system_info("llama-cpp", "b1234")
        filepath = export_json([summary], system_info=si, output_dir=tmp_path)
        assert filepath.name.startswith("benchmark_llama-cpp_")
        assert filepath.suffix == ".json"

    def test_csv_filename_includes_backend(self, tmp_path):
        """CSV filename follows pattern: benchmark_{backend}_{timestamp}.csv."""
        from llm_benchmark.exporters import export_csv

        summary = _make_summary([_make_result()])
        si = _make_system_info("ollama", "0.6.1")
        filepath = export_csv([summary], system_info=si, output_dir=tmp_path)
        assert filepath.name.startswith("benchmark_ollama_")
        assert filepath.suffix == ".csv"

    def test_markdown_filename_includes_backend(self, tmp_path):
        """Markdown filename follows pattern: benchmark_{backend}_{timestamp}.md."""
        from llm_benchmark.exporters import export_markdown

        summary = _make_summary([_make_result()])
        si = _make_system_info("lm-studio", "0.3.0")
        filepath = export_markdown([summary], system_info=si, output_dir=tmp_path)
        assert filepath.name.startswith("benchmark_lm-studio_")
        assert filepath.suffix == ".md"

    def test_filename_without_system_info_omits_backend(self, tmp_path):
        """Without system_info, filename falls back to benchmark_{timestamp}."""
        from llm_benchmark.exporters import export_json

        summary = _make_summary([_make_result()])
        filepath = export_json([summary], system_info=None, output_dir=tmp_path)
        # Should NOT have a backend segment -- just benchmark_ then timestamp
        name = filepath.stem  # e.g. "benchmark_20260314_120000"
        parts = name.split("_")
        assert parts[0] == "benchmark"
        # Timestamp part: YYYYMMDD
        assert len(parts[1]) == 8


class TestJsonMetadata:
    """Test that JSON output includes backend_name in metadata."""

    def test_json_contains_backend_name(self, tmp_path):
        """JSON metadata includes backend_name field."""
        from llm_benchmark.exporters import export_json

        summary = _make_summary([_make_result()])
        si = _make_system_info("llama-cpp", "b1234")
        filepath = export_json([summary], system_info=si, output_dir=tmp_path)
        data = json.loads(filepath.read_text())
        assert data["system_info"]["backend_name"] == "llama-cpp"
        assert data["system_info"]["backend_version"] == "b1234"

    def test_json_contains_backend_version(self, tmp_path):
        """JSON metadata includes backend_version field."""
        from llm_benchmark.exporters import export_json

        summary = _make_summary([_make_result()])
        si = _make_system_info("ollama", "0.6.1")
        filepath = export_json([summary], system_info=si, output_dir=tmp_path)
        data = json.loads(filepath.read_text())
        assert data["system_info"]["backend_version"] == "0.6.1"


class TestMarkdownBackendHeader:
    """Test that Markdown report header shows backend name."""

    def test_markdown_header_includes_backend(self, tmp_path):
        """Markdown report title includes backend name in parentheses."""
        from llm_benchmark.exporters import export_markdown

        summary = _make_summary([_make_result()])
        si = _make_system_info("llama-cpp", "b1234")
        filepath = export_markdown([summary], system_info=si, output_dir=tmp_path)
        content = filepath.read_text()
        assert "# LLM Benchmark Results (llama-cpp)" in content

    def test_markdown_header_has_backend_field(self, tmp_path):
        """Markdown header line includes **Backend:** field."""
        from llm_benchmark.exporters import export_markdown

        summary = _make_summary([_make_result()])
        si = _make_system_info("ollama", "0.6.1")
        filepath = export_markdown([summary], system_info=si, output_dir=tmp_path)
        content = filepath.read_text()
        assert "**Backend:** ollama 0.6.1" in content

    def test_markdown_without_system_info_no_backend(self, tmp_path):
        """Without system_info, Markdown title has no backend label."""
        from llm_benchmark.exporters import export_markdown

        summary = _make_summary([_make_result()])
        filepath = export_markdown([summary], system_info=None, output_dir=tmp_path)
        content = filepath.read_text()
        assert "# LLM Benchmark Results\n" in content


class TestKnownIssuesHints:
    """Test the known-issues hint table in runner."""

    def test_hint_for_ollama_timeout(self):
        from llm_benchmark.runner import get_known_issue_hint

        hint = get_known_issue_hint("ollama", "Timeout after 200s")
        assert hint is not None
        assert "timeout" in hint.lower() or "smaller" in hint.lower()

    def test_hint_for_llama_cpp_connection(self):
        from llm_benchmark.runner import get_known_issue_hint

        hint = get_known_issue_hint("llama-cpp", "Connection refused")
        assert hint is not None
        assert "llama-server" in hint

    def test_no_hint_for_unknown_pattern(self):
        from llm_benchmark.runner import get_known_issue_hint

        hint = get_known_issue_hint("ollama", "some random error")
        assert hint is None
