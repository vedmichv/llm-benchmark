"""Tests for the analyze subcommand and compare enhancements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_sample_results(models_data: list[dict]) -> dict:
    """Build a minimal benchmark results dict."""
    return {
        "generated": "2026-03-12T12:00:00Z",
        "system_info": {
            "cpu": "Test CPU",
            "ram_gb": 16.0,
            "gpu": "Test GPU",
            "backend_name": "ollama",
            "backend_version": "0.6.0",
        },
        "models": models_data,
    }


def _make_model(
    name: str,
    response_ts: float,
    total_ts: float,
    prompt_eval_ts: float,
    load_durations: list[float] | None = None,
) -> dict:
    """Build a model entry with averages and runs."""
    if load_durations is None:
        load_durations = [0.5, 0.6]
    runs = []
    for i, ld in enumerate(load_durations):
        runs.append({
            "model": name,
            "prompt": f"prompt {i}",
            "success": True,
            "eval_count": 100 + i * 10,
            "eval_duration_s": 2.0 + i * 0.1,
            "total_duration_s": 3.0 + i * 0.1,
            "load_duration_s": ld,
            "response_ts": response_ts + i * 0.5,
            "prompt_eval_ts": prompt_eval_ts + i * 0.3,
            "prompt_cached": False,
        })
    return {
        "model": name,
        "averages": {
            "response_ts": response_ts,
            "total_ts": total_ts,
            "prompt_eval_ts": prompt_eval_ts,
        },
        "runs": runs,
    }


@pytest.fixture
def sample_results_json(tmp_path: Path) -> Path:
    """Create a sample JSON results file with 3 models."""
    data = _make_sample_results([
        _make_model("fast-model:1b", response_ts=80.0, total_ts=70.0, prompt_eval_ts=200.0, load_durations=[0.3, 0.4]),
        _make_model("mid-model:3b", response_ts=50.0, total_ts=45.0, prompt_eval_ts=150.0, load_durations=[0.8, 1.0]),
        _make_model("slow-model:7b", response_ts=20.0, total_ts=18.0, prompt_eval_ts=80.0, load_durations=[2.0, 2.5]),
    ])
    fp = tmp_path / "results.json"
    fp.write_text(json.dumps(data))
    return fp


# ── Task 1: Analyze tests ──────────────────────────────────────────


class TestGetSortValue:
    """Test _get_sort_value helper."""

    def test_response_ts(self) -> None:
        from llm_benchmark.analyze import _get_sort_value

        model = _make_model("m", 45.0, 40.0, 100.0)
        assert _get_sort_value(model, "response_ts") == 45.0

    def test_total_ts(self) -> None:
        from llm_benchmark.analyze import _get_sort_value

        model = _make_model("m", 45.0, 40.0, 100.0)
        assert _get_sort_value(model, "total_ts") == 40.0

    def test_load_time(self) -> None:
        from llm_benchmark.analyze import _get_sort_value

        model = _make_model("m", 45.0, 40.0, 100.0, load_durations=[1.0, 3.0])
        val = _get_sort_value(model, "load_time")
        assert val == pytest.approx(2.0)

    def test_missing_key_returns_zero(self) -> None:
        from llm_benchmark.analyze import _get_sort_value

        model = {"model": "m", "averages": {}, "runs": []}
        assert _get_sort_value(model, "response_ts") == 0


class TestAnalyzeResults:
    """Test analyze_results function."""

    def test_sort_by_response_ts(self, sample_results_json: Path) -> None:
        from llm_benchmark.analyze import _get_sort_value, analyze_results

        # Load and check sort order
        data = json.loads(sample_results_json.read_text())
        models = data["models"]
        sorted_models = sorted(
            models, key=lambda m: _get_sort_value(m, "response_ts"), reverse=True
        )
        assert sorted_models[0]["model"] == "fast-model:1b"

        # Function should not crash
        analyze_results(str(sample_results_json))

    def test_sort_ascending(self, sample_results_json: Path) -> None:
        from llm_benchmark.analyze import _get_sort_value

        data = json.loads(sample_results_json.read_text())
        models = data["models"]
        sorted_models = sorted(
            models, key=lambda m: _get_sort_value(m, "response_ts"), reverse=False
        )
        assert sorted_models[0]["model"] == "slow-model:7b"

        from llm_benchmark.analyze import analyze_results
        analyze_results(str(sample_results_json), ascending=True)

    def test_top_n_filter(self, sample_results_json: Path) -> None:
        from llm_benchmark.analyze import analyze_results

        # Should not crash, should only show 2 models
        analyze_results(str(sample_results_json), top_n=2)

    def test_sort_by_load_time(self, sample_results_json: Path) -> None:
        from llm_benchmark.analyze import _get_sort_value

        data = json.loads(sample_results_json.read_text())
        models = data["models"]
        sorted_models = sorted(
            models, key=lambda m: _get_sort_value(m, "load_time"), reverse=True
        )
        # slow-model has highest load_duration_s (avg 2.25)
        assert sorted_models[0]["model"] == "slow-model:7b"

        from llm_benchmark.analyze import analyze_results
        analyze_results(str(sample_results_json), sort_by="load_time")

    def test_invalid_sort_key(self, sample_results_json: Path) -> None:
        from llm_benchmark.analyze import analyze_results

        # Should not crash -- prints error gracefully
        analyze_results(str(sample_results_json), sort_by="invalid")

    def test_file_not_found(self) -> None:
        from llm_benchmark.analyze import analyze_results

        # Should not crash
        analyze_results("/nonexistent/file.json")

    def test_detail_mode(self, sample_results_json: Path) -> None:
        from llm_benchmark.analyze import analyze_results

        # Should not crash
        analyze_results(str(sample_results_json), detail=True)


# ── Task 2: Compare enhancement tests ──────────────────────────────


def _write_results_file(tmp_path: Path, name: str, models_data: list[dict]) -> Path:
    """Write a results JSON file to tmp_path."""
    fp = tmp_path / name
    fp.write_text(json.dumps(_make_sample_results(models_data)))
    return fp


class TestCompareArrows:
    """Test enhanced compare with arrows and winner column."""

    def test_compare_arrows(self, tmp_path: Path) -> None:
        """Compare two files; verify arrows don't crash."""
        from llm_benchmark.compare import compare_results

        f1 = _write_results_file(tmp_path, "run1.json", [
            _make_model("model-a:1b", response_ts=40.0, total_ts=35.0, prompt_eval_ts=100.0),
        ])
        f2 = _write_results_file(tmp_path, "run2.json", [
            _make_model("model-a:1b", response_ts=50.0, total_ts=42.0, prompt_eval_ts=120.0),
        ])
        # Should complete without exception and show arrows
        compare_results([str(f1), str(f2)])

    def test_compare_backward_compat(self, tmp_path: Path) -> None:
        """JSON without 'mode' field should not crash."""
        from llm_benchmark.compare import compare_results

        # Explicitly no "mode" field in results
        data1 = _make_sample_results([
            _make_model("model-a:1b", response_ts=40.0, total_ts=35.0, prompt_eval_ts=100.0),
        ])
        data2 = _make_sample_results([
            _make_model("model-a:1b", response_ts=45.0, total_ts=38.0, prompt_eval_ts=110.0),
        ])
        # Ensure no "mode" key anywhere
        assert "mode" not in data1
        assert "mode" not in data2

        f1 = tmp_path / "compat1.json"
        f2 = tmp_path / "compat2.json"
        f1.write_text(json.dumps(data1))
        f2.write_text(json.dumps(data2))

        compare_results([str(f1), str(f2)])

    def test_compare_winner(self, tmp_path: Path) -> None:
        """Run2 is clearly faster; compare completes with winner info."""
        from llm_benchmark.compare import compare_results

        f1 = _write_results_file(tmp_path, "slow.json", [
            _make_model("model-a:1b", response_ts=20.0, total_ts=18.0, prompt_eval_ts=50.0),
        ])
        f2 = _write_results_file(tmp_path, "fast.json", [
            _make_model("model-a:1b", response_ts=60.0, total_ts=55.0, prompt_eval_ts=150.0),
        ])
        compare_results([str(f1), str(f2)])
