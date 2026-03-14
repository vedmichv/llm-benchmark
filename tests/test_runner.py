"""Tests for llm_benchmark.runner module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from llm_benchmark.backends import BackendError, BackendResponse
from llm_benchmark.config import DEFAULT_TIMEOUT
from llm_benchmark.models import BenchmarkResult


def _make_backend_response(
    prompt_eval_count: int = 15,
    prompt_eval_duration: float = 0.2,
    eval_count: int = 120,
    eval_duration: float = 4.0,
    prompt_cached: bool = False,
) -> BackendResponse:
    """Helper to build a BackendResponse with seconds-based timing."""
    return BackendResponse(
        model="test-model",
        content="test response",
        done=True,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        total_duration=prompt_eval_duration + eval_duration,
        load_duration=0.0,
        prompt_eval_duration=prompt_eval_duration,
        eval_duration=eval_duration,
        prompt_cached=prompt_cached,
    )


def _make_mock_backend(**kwargs) -> MagicMock:
    """Create a mock Backend with default behavior."""
    backend = MagicMock()
    backend.name = "ollama"
    backend.version = "0.6.1"
    backend.warmup.return_value = True
    backend.unload_model.return_value = True
    backend.detect_context_window.return_value = 4096
    backend.get_model_size.return_value = 2.0
    backend.check_connectivity.return_value = True
    for k, v in kwargs.items():
        setattr(backend, k, v)
    return backend


class TestComputeAveragesRunner:
    """Test correct averaging imported via runner (delegates to models.compute_averages)."""

    def _make_result(
        self,
        prompt_eval_count: int,
        prompt_eval_duration: float,
        eval_count: int,
        eval_duration: float,
        prompt_cached: bool = False,
    ) -> BenchmarkResult:
        """Helper to build a BenchmarkResult with a valid BackendResponse."""
        resp = _make_backend_response(
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
            prompt_eval_duration=1.0,
            eval_count=100,
            eval_duration=2.0,
        )
        r2 = self._make_result(
            prompt_eval_count=20,
            prompt_eval_duration=2.0,
            eval_count=200,
            eval_duration=5.0,
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
            prompt_eval_duration=1.0,
            eval_count=100,
            eval_duration=2.0,
        )
        r_cached = self._make_result(
            prompt_eval_count=0,
            prompt_eval_duration=0.0,
            eval_count=200,
            eval_duration=5.0,
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

    def test_unload_model_calls_backend(self):
        """Verify unload_model calls backend.unload_model."""
        from llm_benchmark.runner import unload_model

        backend = _make_mock_backend()
        result = unload_model(backend, "llama3.2:1b")

        backend.unload_model.assert_called_once_with("llama3.2:1b")
        assert result is True

    def test_unload_model_returns_false_on_error(self):
        """Verify unload_model returns False when backend raises."""
        from llm_benchmark.runner import unload_model

        backend = _make_mock_backend()
        backend.unload_model.side_effect = Exception("connection error")
        result = unload_model(backend, "llama3.2:1b")
        assert result is False


class TestRunSingleBenchmark:
    """Test single benchmark execution."""

    def test_success(self):
        """Verify run_single_benchmark returns BenchmarkResult with success=True."""
        from llm_benchmark.runner import run_single_benchmark

        backend = _make_mock_backend()
        backend.chat.return_value = _make_backend_response()

        result = run_single_benchmark(backend, "llama3.2:1b", "Why is the sky blue?")

        assert result.success is True
        assert result.model == "llama3.2:1b"
        assert result.response is not None
        assert result.error is None

    def test_failure(self):
        """Verify run_single_benchmark returns BenchmarkResult with success=False on error."""
        from llm_benchmark.runner import run_single_benchmark

        backend = _make_mock_backend()
        backend.chat.side_effect = Exception("model not found")

        result = run_single_benchmark(backend, "nonexistent:latest", "test prompt")

        assert result.success is False
        assert result.error is not None
        assert "model not found" in result.error


class TestWarmupModel:
    """Test warmup_model() function."""

    def test_warmup_calls_backend_warmup(self):
        """warmup_model() calls backend.warmup with model name and timeout."""
        from llm_benchmark.runner import warmup_model

        backend = _make_mock_backend()
        warmup_model(backend, "llama3.2:1b")

        backend.warmup.assert_called_once()

    def test_warmup_returns_true_on_success(self):
        """warmup_model() returns True on success."""
        from llm_benchmark.runner import warmup_model

        backend = _make_mock_backend()
        result = warmup_model(backend, "llama3.2:1b")
        assert result is True

    def test_warmup_returns_false_on_exception(self):
        """warmup_model() returns False on exception."""
        from llm_benchmark.runner import warmup_model

        backend = _make_mock_backend()
        backend.warmup.side_effect = Exception("connection refused")
        result = warmup_model(backend, "llama3.2:1b")
        assert result is False

    def test_warmup_does_not_raise_on_failure(self):
        """warmup_model() does not raise on exception."""
        from llm_benchmark.runner import warmup_model

        backend = _make_mock_backend()
        backend.warmup.side_effect = ConnectionError("refused")
        # Should NOT raise
        result = warmup_model(backend, "llama3.2:1b")
        assert result is False


class TestRetryLogic:
    """Test retry logic in run_single_benchmark."""

    def test_retries_on_backend_error_retryable(self):
        """run_single_benchmark retries on BackendError with retryable=True."""
        from llm_benchmark.runner import run_single_benchmark

        backend = _make_mock_backend()
        backend.chat.side_effect = [
            BackendError("connection refused", retryable=True),
            _make_backend_response(),
        ]

        result = run_single_benchmark(backend, "llama3.2:1b", "test", max_retries=3)
        assert result.success is True
        assert backend.chat.call_count == 2

    @patch("llm_benchmark.runner.unload_model")
    def test_no_retry_on_timeout_error(self, mock_unload):
        """run_single_benchmark does NOT retry on TimeoutError (wastes minutes)."""
        from llm_benchmark.runner import run_single_benchmark

        backend = _make_mock_backend()
        backend.chat.side_effect = TimeoutError("timed out")

        result = run_single_benchmark(backend, "llama3.2:1b", "test", max_retries=3)
        assert result.success is False
        assert "Timeout" in result.error

    def test_no_retry_on_backend_error_not_retryable(self):
        """run_single_benchmark does NOT retry on BackendError with retryable=False."""
        from llm_benchmark.runner import run_single_benchmark

        backend = _make_mock_backend()
        backend.chat.side_effect = BackendError("not found", retryable=False)

        result = run_single_benchmark(backend, "llama3.2:1b", "test", max_retries=3)
        assert result.success is False
        # Should only be called once (no retry)
        assert backend.chat.call_count == 1

    def test_max_retries_zero_no_retry(self):
        """run_single_benchmark with max_retries=0 does not retry."""
        from llm_benchmark.runner import run_single_benchmark

        backend = _make_mock_backend()
        backend.chat.side_effect = BackendError("refused", retryable=True)

        result = run_single_benchmark(backend, "llama3.2:1b", "test", max_retries=0)
        assert result.success is False
        assert backend.chat.call_count == 1

    def test_retries_exhausted_returns_failure(self):
        """After retries exhausted, returns BenchmarkResult(success=False) with error."""
        from llm_benchmark.runner import run_single_benchmark

        backend = _make_mock_backend()
        backend.chat.side_effect = BackendError("refused", retryable=True)

        result = run_single_benchmark(backend, "llama3.2:1b", "test", max_retries=2)
        assert result.success is False
        assert result.error is not None
        # Should have tried 1 initial + 2 retries = 3 calls
        assert backend.chat.call_count == 3


class TestCacheVisibility:
    """Test cache visibility indicators in benchmark_model terminal output."""

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_cached_result_shows_cached_tag(self, mock_run, mock_warmup):
        """When result.prompt_cached is True, '[cached]' appears in console output."""
        import io

        from rich.console import Console

        from llm_benchmark.runner import benchmark_model

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)

        mock_run.return_value = BenchmarkResult(
            model="test-model",
            prompt="test prompt",
            success=True,
            response=_make_backend_response(prompt_eval_count=0, prompt_eval_duration=0.0, prompt_cached=True),
            prompt_cached=True,
        )

        backend = _make_mock_backend()
        with patch("llm_benchmark.runner.get_console", return_value=test_console):
            benchmark_model(backend, "test-model", ["test prompt"], skip_warmup=True)

        output = buf.getvalue()
        assert "[cached]" in output

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_first_cached_shows_explanation(self, mock_run, mock_warmup):
        """First cached result in session triggers one-liner explanation."""
        import io

        from rich.console import Console

        from llm_benchmark.runner import benchmark_model

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)

        mock_run.return_value = BenchmarkResult(
            model="test-model",
            prompt="test prompt",
            success=True,
            response=_make_backend_response(prompt_eval_count=0, prompt_eval_duration=0.0, prompt_cached=True),
            prompt_cached=True,
        )

        backend = _make_mock_backend()
        with patch("llm_benchmark.runner.get_console", return_value=test_console):
            benchmark_model(backend, "test-model", ["test prompt"], skip_warmup=True)

        output = buf.getvalue()
        assert "Prompt caching" in output

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_second_cached_no_repeat_explanation(self, mock_run, mock_warmup):
        """Second cached result does NOT repeat the explanation."""
        import io

        from rich.console import Console

        from llm_benchmark.runner import benchmark_model

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)

        mock_run.return_value = BenchmarkResult(
            model="test-model",
            prompt="test prompt",
            success=True,
            response=_make_backend_response(prompt_eval_count=0, prompt_eval_duration=0.0, prompt_cached=True),
            prompt_cached=True,
        )

        backend = _make_mock_backend()
        with patch("llm_benchmark.runner.get_console", return_value=test_console):
            benchmark_model(backend, "test-model", ["prompt one", "prompt two"], skip_warmup=True)

        output = buf.getvalue()
        # "Prompt caching" should appear exactly once
        assert output.count("Prompt caching") == 1

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_all_cached_shows_warning(self, mock_run, mock_warmup):
        """When ALL results for a model are cached, warning about unavailable prompt eval metrics is shown."""
        import io

        from rich.console import Console

        from llm_benchmark.runner import benchmark_model

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)

        mock_run.return_value = BenchmarkResult(
            model="test-model",
            prompt="test prompt",
            success=True,
            response=_make_backend_response(prompt_eval_count=0, prompt_eval_duration=0.0, prompt_cached=True),
            prompt_cached=True,
        )

        backend = _make_mock_backend()
        with patch("llm_benchmark.runner.get_console", return_value=test_console):
            benchmark_model(backend, "test-model", ["prompt one", "prompt two"], skip_warmup=True)

        output = buf.getvalue()
        assert "prompt eval metrics unavailable" in output.lower() or "All runs cached" in output

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_non_cached_no_tag(self, mock_run, mock_warmup):
        """Non-cached results do NOT show [cached] tag."""
        import io

        from rich.console import Console

        from llm_benchmark.runner import benchmark_model

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True)

        mock_run.return_value = BenchmarkResult(
            model="test-model",
            prompt="test prompt",
            success=True,
            response=_make_backend_response(),
            prompt_cached=False,
        )

        backend = _make_mock_backend()
        with patch("llm_benchmark.runner.get_console", return_value=test_console):
            benchmark_model(backend, "test-model", ["test prompt"], skip_warmup=True)

        output = buf.getvalue()
        assert "[cached]" not in output


class TestBenchmarkModelWarmup:
    """Test warmup integration in benchmark_model."""

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_warmup_called_when_not_skipped(self, mock_run, mock_warmup):
        """benchmark_model calls warmup_model before prompt loop when skip_warmup=False."""
        from llm_benchmark.runner import benchmark_model

        mock_run.return_value = BenchmarkResult(
            model="llama3.2:1b",
            prompt="test",
            success=True,
            response=_make_backend_response(),
        )

        backend = _make_mock_backend()
        benchmark_model(backend, "llama3.2:1b", ["test prompt"], skip_warmup=False)

        mock_warmup.assert_called_once_with(backend, "llama3.2:1b", DEFAULT_TIMEOUT)

    @patch("llm_benchmark.runner.warmup_model", return_value=True)
    @patch("llm_benchmark.runner.run_single_benchmark")
    def test_warmup_skipped_when_flag_set(self, mock_run, mock_warmup):
        """benchmark_model skips warmup when skip_warmup=True."""
        from llm_benchmark.runner import benchmark_model

        mock_run.return_value = BenchmarkResult(
            model="llama3.2:1b",
            prompt="test",
            success=True,
            response=_make_backend_response(),
        )

        backend = _make_mock_backend()
        benchmark_model(backend, "llama3.2:1b", ["test prompt"], skip_warmup=True)

        mock_warmup.assert_not_called()
