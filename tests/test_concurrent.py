"""Tests for concurrent benchmarking module."""


from llm_benchmark.backends import BackendResponse
from llm_benchmark.concurrent import (
    auto_detect_concurrency,
    run_concurrent_batch,
)
from llm_benchmark.config import DEFAULT_CONCURRENT, SWEEP_NUM_CTX, SWEEP_PROMPT
from llm_benchmark.models import (
    ConcurrentBatchResult,
    SweepConfigResult,
    SweepModelResult,
)


class TestConcurrentBatchResultModel:
    """Test ConcurrentBatchResult Pydantic model construction."""

    def test_concurrent_batch_result_model(self, sample_benchmark_result):
        """ConcurrentBatchResult validates with all required fields."""
        batch = ConcurrentBatchResult(
            model="llama3.2:1b",
            prompt="Why is the sky blue?",
            num_workers=4,
            wall_time_s=2.5,
            results=[sample_benchmark_result],
            aggregate_throughput_ts=48.0,
            avg_request_throughput_ts=30.0,
        )
        assert batch.model == "llama3.2:1b"
        assert batch.num_workers == 4
        assert batch.wall_time_s == 2.5
        assert len(batch.results) == 1
        assert batch.aggregate_throughput_ts == 48.0
        assert batch.avg_request_throughput_ts == 30.0

    def test_concurrent_batch_result_empty_results(self):
        """ConcurrentBatchResult accepts empty results list."""
        batch = ConcurrentBatchResult(
            model="llama3.2:1b",
            prompt="Hello",
            num_workers=2,
            wall_time_s=0.1,
            results=[],
            aggregate_throughput_ts=0.0,
            avg_request_throughput_ts=0.0,
        )
        assert batch.results == []


class TestSweepModels:
    """Test SweepConfigResult and SweepModelResult models."""

    def test_sweep_config_result_success(self):
        """SweepConfigResult validates for a successful sweep config."""
        cfg = SweepConfigResult(
            model="llama3.2:1b",
            num_ctx=2048,
            num_gpu=1,
            response_ts=35.0,
            total_ts=30.0,
            eval_count=150,
            total_duration_s=5.0,
            success=True,
            error=None,
        )
        assert cfg.success is True
        assert cfg.num_ctx == 2048
        assert cfg.error is None

    def test_sweep_config_result_failure(self):
        """SweepConfigResult validates for a failed config."""
        cfg = SweepConfigResult(
            model="llama3.2:1b",
            num_ctx=4096,
            num_gpu=0,
            response_ts=0.0,
            total_ts=0.0,
            eval_count=0,
            total_duration_s=0.0,
            success=False,
            error="Out of memory",
        )
        assert cfg.success is False
        assert cfg.error == "Out of memory"

    def test_sweep_model_result_with_best(self):
        """SweepModelResult with best_config set."""
        best = SweepConfigResult(
            model="llama3.2:1b",
            num_ctx=2048,
            num_gpu=1,
            response_ts=35.0,
            total_ts=30.0,
            eval_count=150,
            total_duration_s=5.0,
            success=True,
            error=None,
        )
        sweep = SweepModelResult(
            model="llama3.2:1b",
            configs=[best],
            best_config=best,
        )
        assert sweep.best_config is not None
        assert sweep.best_config.num_ctx == 2048

    def test_sweep_model_result_no_best(self):
        """SweepModelResult with no best config (all failed)."""
        sweep = SweepModelResult(
            model="llama3.2:1b",
            configs=[],
            best_config=None,
        )
        assert sweep.best_config is None


class TestConfigConstants:
    """Test that Phase 3 config constants exist and have correct values."""

    def test_default_concurrent(self):
        assert DEFAULT_CONCURRENT == 4

    def test_sweep_num_ctx(self):
        assert SWEEP_NUM_CTX == [512, 1024, 2048, 4096]

    def test_sweep_prompt(self):
        assert isinstance(SWEEP_PROMPT, str)
        assert len(SWEEP_PROMPT) > 10


class TestAutoDetectConcurrency:
    """Test auto_detect_concurrency resource-based defaults."""

    def test_high_vram_returns_8(self):
        assert auto_detect_concurrency(64.0, 16.0) == 8

    def test_high_vram_boundary(self):
        assert auto_detect_concurrency(16.0, 16.0) == 8

    def test_high_ram_no_gpu_returns_4(self):
        assert auto_detect_concurrency(32.0, None) == 4

    def test_high_ram_low_gpu_returns_4(self):
        assert auto_detect_concurrency(32.0, 8.0) == 4

    def test_low_resources_returns_2(self):
        assert auto_detect_concurrency(16.0, None) == 2

    def test_low_ram_low_gpu_returns_2(self):
        assert auto_detect_concurrency(16.0, 8.0) == 2


class TestRunConcurrentBatch:
    """Test run_concurrent_batch fires N parallel requests via ThreadPoolExecutor."""

    def test_run_concurrent_batch_returns_n_results(
        self, mock_backend, sample_backend_response
    ):
        """N workers produce N results with correct wall time."""
        batch = run_concurrent_batch(mock_backend, "llama3.2:1b", "Hello", n=3, timeout=30)

        assert isinstance(batch, ConcurrentBatchResult)
        assert len(batch.results) == 3
        assert all(r.success for r in batch.results)
        assert batch.wall_time_s > 0
        assert batch.num_workers == 3

    def test_failed_request_continues(self, mock_backend):
        """One failing request does not stop others."""
        call_count = 0

        def chat_side_effect(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection refused")
            return BackendResponse(
                model=model,
                content="test",
                done=True,
                eval_count=120,
                eval_duration=4.0,
                total_duration=5.0,
            )

        mock_backend.chat.side_effect = chat_side_effect

        batch = run_concurrent_batch(mock_backend, "llama3.2:1b", "Hello", n=3, timeout=30)

        assert len(batch.results) == 3
        failed = [r for r in batch.results if not r.success]
        succeeded = [r for r in batch.results if r.success]
        assert len(failed) == 1
        assert len(succeeded) == 2
        assert "Connection refused" in failed[0].error

    def test_aggregate_throughput_calculation(
        self, mock_backend, sample_backend_response
    ):
        """aggregate_throughput_ts = sum(eval_count) / wall_time_s."""
        batch = run_concurrent_batch(mock_backend, "llama3.2:1b", "Hello", n=2, timeout=30)

        # Each response has eval_count=120, so total=240
        # aggregate_throughput_ts = 240 / wall_time_s
        expected = 240.0 / batch.wall_time_s
        assert abs(batch.aggregate_throughput_ts - expected) < 0.1
