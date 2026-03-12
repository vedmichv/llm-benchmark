"""Tests for llm_benchmark.runner module."""

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from llm_benchmark.models import BenchmarkResult, OllamaResponse, ModelSummary


class TestComputeAveragesRunner:
    """Test correct averaging imported via runner (delegates to models.compute_averages)."""

    def _make_result(
        self,
        prompt_eval_count: int,
        prompt_eval_duration: int,
        eval_count: int,
        eval_duration: int,
        prompt_cached: bool = False,
    ) -> BenchmarkResult:
        """Helper to build a BenchmarkResult with a valid OllamaResponse."""
        resp = OllamaResponse(
            model="test-model",
            created_at=datetime.now(timezone.utc),
            message={"role": "assistant", "content": "test"},
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
            prompt="test prompt",
            success=True,
            response=resp,
            prompt_cached=prompt_cached,
        )

    def test_correct_averaging(self):
        """Verify compute_averages returns sum_tokens/sum_time, not mean of rates.

        Example from RESEARCH.md:
        Run 1: 100 tokens in 2s = 50 t/s
        Run 2: 200 tokens in 5s = 40 t/s
        Arithmetic mean of rates: (50+40)/2 = 45 t/s  (WRONG)
        Correct: 300 tokens / 7s = 42.86 t/s
        """
        from llm_benchmark.runner import compute_averages

        r1 = self._make_result(
            prompt_eval_count=10,
            prompt_eval_duration=1_000_000_000,  # 1s
            eval_count=100,
            eval_duration=2_000_000_000,  # 2s
        )
        r2 = self._make_result(
            prompt_eval_count=20,
            prompt_eval_duration=2_000_000_000,  # 2s
            eval_count=200,
            eval_duration=5_000_000_000,  # 5s
        )

        avgs = compute_averages([r1, r2])

        # Response: (100+200) / (2+5) = 300/7 = 42.857...
        assert abs(avgs["response_ts"] - 300 / 7) < 0.01
        # Prompt eval: (10+20) / (1+2) = 30/3 = 10.0
        assert abs(avgs["prompt_eval_ts"] - 10.0) < 0.01
        # Total: (10+20+100+200) / (1+2+2+5) = 330/10 = 33.0
        assert abs(avgs["total_ts"] - 33.0) < 0.01

    def test_correct_averaging_excludes_cached(self):
        """Verify prompt_eval average excludes prompt-cached results."""
        from llm_benchmark.runner import compute_averages

        r_normal = self._make_result(
            prompt_eval_count=10,
            prompt_eval_duration=1_000_000_000,
            eval_count=100,
            eval_duration=2_000_000_000,
        )
        r_cached = self._make_result(
            prompt_eval_count=0,
            prompt_eval_duration=0,
            eval_count=200,
            eval_duration=5_000_000_000,
            prompt_cached=True,
        )

        avgs = compute_averages([r_normal, r_cached])

        # Prompt eval should only count from r_normal: 10/1 = 10.0
        assert abs(avgs["prompt_eval_ts"] - 10.0) < 0.01
        # Response should include both: (100+200)/(2+5) = 42.857
        assert abs(avgs["response_ts"] - 300 / 7) < 0.01


class TestRunWithTimeout:
    """Test threading-based timeout (STAB-06)."""

    def test_timeout_raises(self):
        """Verify run_with_timeout raises TimeoutError for slow functions."""
        from llm_benchmark.runner import run_with_timeout

        def slow_func():
            time.sleep(10)
            return "done"

        with pytest.raises(TimeoutError):
            run_with_timeout(slow_func, timeout_seconds=0.2)

    def test_timeout_returns_result(self):
        """Verify run_with_timeout returns result for fast functions."""
        from llm_benchmark.runner import run_with_timeout

        def fast_func():
            return 42

        result = run_with_timeout(fast_func, timeout_seconds=5)
        assert result == 42

    def test_timeout_no_sigalrm(self):
        """Confirm no signal.SIGALRM usage in runner module."""
        import inspect
        import llm_benchmark.runner as runner_mod

        source = inspect.getsource(runner_mod)
        assert "SIGALRM" not in source


class TestUnloadModel:
    """Test model offloading (STAB-05)."""

    @patch("llm_benchmark.runner.ollama")
    def test_unload_model_calls_keep_alive_zero(self, mock_ollama):
        """Verify unload_model calls ollama.generate with keep_alive=0."""
        from llm_benchmark.runner import unload_model

        result = unload_model("llama3.2:1b")

        mock_ollama.generate.assert_called_once_with(
            model="llama3.2:1b", prompt="", keep_alive=0
        )
        assert result is True

    @patch("llm_benchmark.runner.ollama")
    def test_unload_model_returns_false_on_error(self, mock_ollama):
        """Verify unload_model returns False when ollama.generate raises."""
        from llm_benchmark.runner import unload_model

        mock_ollama.generate.side_effect = Exception("connection error")
        result = unload_model("llama3.2:1b")
        assert result is False


class TestRunSingleBenchmark:
    """Test single benchmark execution."""

    @patch("llm_benchmark.runner.ollama")
    def test_success(self, mock_ollama, sample_ollama_response_dict):
        """Verify run_single_benchmark returns BenchmarkResult with success=True."""
        from llm_benchmark.runner import run_single_benchmark

        # Mock ollama.chat to return a dict-like response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = sample_ollama_response_dict
        mock_ollama.chat.return_value = mock_response

        result = run_single_benchmark("llama3.2:1b", "Why is the sky blue?")

        assert result.success is True
        assert result.model == "llama3.2:1b"
        assert result.response is not None
        assert result.error is None

    @patch("llm_benchmark.runner.ollama")
    def test_failure(self, mock_ollama):
        """Verify run_single_benchmark returns BenchmarkResult with success=False on error."""
        from llm_benchmark.runner import run_single_benchmark

        mock_ollama.chat.side_effect = Exception("model not found")

        result = run_single_benchmark("nonexistent:latest", "test prompt")

        assert result.success is False
        assert result.error is not None
        assert "model not found" in result.error
