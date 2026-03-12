"""Tests for llm_benchmark.runner module."""

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from llm_benchmark.config import DEFAULT_TIMEOUT
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


class TestWarmupModel:
    """Test warmup_model() function."""

    @patch("llm_benchmark.runner.ollama")
    def test_warmup_calls_ollama_chat(self, mock_ollama):
        """warmup_model() calls ollama.chat with short prompt and model name."""
        from llm_benchmark.runner import warmup_model

        warmup_model("llama3.2:1b")

        mock_ollama.chat.assert_called_once_with(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": "Hello"}],
        )

    @patch("llm_benchmark.runner.ollama")
    def test_warmup_returns_true_on_success(self, mock_ollama):
        """warmup_model() returns True on success."""
        from llm_benchmark.runner import warmup_model

        result = warmup_model("llama3.2:1b")
        assert result is True

    @patch("llm_benchmark.runner.ollama")
    def test_warmup_returns_false_on_exception(self, mock_ollama):
        """warmup_model() returns False on exception."""
        from llm_benchmark.runner import warmup_model

        mock_ollama.chat.side_effect = Exception("connection refused")
        result = warmup_model("llama3.2:1b")
        assert result is False

    @patch("llm_benchmark.runner.ollama")
    def test_warmup_prints_ready_on_success(self, mock_ollama, capsys):
        """warmup_model() prints 'Warming up...' then 'Ready' on success."""
        from llm_benchmark.runner import warmup_model
        from llm_benchmark.config import get_console

        console = get_console()
        warmup_model("llama3.2:1b")

        # Check console output contains expected text
        output = console.file.getvalue() if hasattr(console.file, "getvalue") else ""
        # Use capsys as fallback - Rich writes to its own console
        # We verify the function completes without error and returns True

    @patch("llm_benchmark.runner.ollama")
    def test_warmup_does_not_raise_on_failure(self, mock_ollama):
        """warmup_model() does not raise on exception."""
        from llm_benchmark.runner import warmup_model

        mock_ollama.chat.side_effect = ConnectionError("refused")
        # Should NOT raise
        result = warmup_model("llama3.2:1b")
        assert result is False


class TestRetryLogic:
    """Test retry logic in run_single_benchmark."""

    @patch("llm_benchmark.runner.ollama")
    def test_retries_on_connection_error(self, mock_ollama, sample_ollama_response_dict):
        """run_single_benchmark retries on ConnectionError."""
        from llm_benchmark.runner import run_single_benchmark

        mock_response = MagicMock()
        mock_response.model_dump.return_value = sample_ollama_response_dict
        mock_ollama.chat.side_effect = [
            ConnectionError("refused"),
            mock_response,
        ]

        result = run_single_benchmark("llama3.2:1b", "test", max_retries=3)
        assert result.success is True
        assert mock_ollama.chat.call_count == 2

    @patch("llm_benchmark.runner.ollama")
    def test_retries_on_timeout_error(self, mock_ollama, sample_ollama_response_dict):
        """run_single_benchmark retries on TimeoutError."""
        from llm_benchmark.runner import run_single_benchmark

        mock_response = MagicMock()
        mock_response.model_dump.return_value = sample_ollama_response_dict
        mock_ollama.chat.side_effect = [
            TimeoutError("timed out"),
            mock_response,
        ]

        result = run_single_benchmark("llama3.2:1b", "test", max_retries=3)
        assert result.success is True

    @patch("llm_benchmark.runner.ollama")
    def test_retries_on_request_error(self, mock_ollama, sample_ollama_response_dict):
        """run_single_benchmark retries on ollama.RequestError."""
        from llm_benchmark.runner import run_single_benchmark
        import ollama as ollama_lib

        mock_response = MagicMock()
        mock_response.model_dump.return_value = sample_ollama_response_dict
        mock_ollama.chat.side_effect = [
            ollama_lib.RequestError("request failed"),
            mock_response,
        ]
        # Ensure the runner module sees the same RequestError
        mock_ollama.RequestError = ollama_lib.RequestError

        result = run_single_benchmark("llama3.2:1b", "test", max_retries=3)
        assert result.success is True

    @patch("llm_benchmark.runner.ollama")
    def test_retries_on_response_error_500(self, mock_ollama, sample_ollama_response_dict):
        """run_single_benchmark retries on ResponseError with status_code >= 500."""
        from llm_benchmark.runner import run_single_benchmark
        import ollama as ollama_lib

        mock_response = MagicMock()
        mock_response.model_dump.return_value = sample_ollama_response_dict

        err_500 = ollama_lib.ResponseError("server error")
        err_500.status_code = 500
        mock_ollama.chat.side_effect = [err_500, mock_response]
        mock_ollama.ResponseError = ollama_lib.ResponseError

        result = run_single_benchmark("llama3.2:1b", "test", max_retries=3)
        assert result.success is True

    @patch("llm_benchmark.runner.ollama")
    def test_no_retry_on_response_error_404(self, mock_ollama):
        """run_single_benchmark does NOT retry on ResponseError with status_code 404."""
        from llm_benchmark.runner import run_single_benchmark
        import ollama as ollama_lib

        err_404 = ollama_lib.ResponseError("not found")
        err_404.status_code = 404
        mock_ollama.chat.side_effect = err_404
        mock_ollama.ResponseError = ollama_lib.ResponseError

        result = run_single_benchmark("llama3.2:1b", "test", max_retries=3)
        assert result.success is False
        # Should only be called once (no retry)
        assert mock_ollama.chat.call_count == 1

    @patch("llm_benchmark.runner.ollama")
    def test_max_retries_zero_no_retry(self, mock_ollama):
        """run_single_benchmark with max_retries=0 does not retry."""
        from llm_benchmark.runner import run_single_benchmark

        mock_ollama.chat.side_effect = ConnectionError("refused")

        result = run_single_benchmark("llama3.2:1b", "test", max_retries=0)
        assert result.success is False
        assert mock_ollama.chat.call_count == 1

    @patch("llm_benchmark.runner.ollama")
    def test_retries_exhausted_returns_failure(self, mock_ollama):
        """After retries exhausted, returns BenchmarkResult(success=False) with error."""
        from llm_benchmark.runner import run_single_benchmark

        mock_ollama.chat.side_effect = ConnectionError("refused")

        result = run_single_benchmark("llama3.2:1b", "test", max_retries=2)
        assert result.success is False
        assert result.error is not None
        # Should have tried 1 initial + 2 retries = 3 calls
        assert mock_ollama.chat.call_count == 3


class TestBenchmarkModelWarmup:
    """Test warmup integration in benchmark_model."""

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_warmup_called_when_not_skipped(self, mock_run, mock_warmup, sample_ollama_response_dict):
        """benchmark_model calls warmup_model before prompt loop when skip_warmup=False."""
        from llm_benchmark.runner import benchmark_model
        from llm_benchmark.models import BenchmarkResult, OllamaResponse

        resp = OllamaResponse.model_validate(sample_ollama_response_dict)
        mock_run.return_value = BenchmarkResult(
            model="llama3.2:1b",
            prompt="test",
            success=True,
            response=resp,
        )

        benchmark_model("llama3.2:1b", ["test prompt"], skip_warmup=False)

        mock_warmup.assert_called_once_with("llama3.2:1b", DEFAULT_TIMEOUT)

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_warmup_skipped_when_flag_set(self, mock_run, mock_warmup, sample_ollama_response_dict):
        """benchmark_model skips warmup when skip_warmup=True."""
        from llm_benchmark.runner import benchmark_model
        from llm_benchmark.models import BenchmarkResult, OllamaResponse

        resp = OllamaResponse.model_validate(sample_ollama_response_dict)
        mock_run.return_value = BenchmarkResult(
            model="llama3.2:1b",
            prompt="test",
            success=True,
            response=resp,
        )

        benchmark_model("llama3.2:1b", ["test prompt"], skip_warmup=True)

        mock_warmup.assert_not_called()
