"""Tests for concurrent benchmarking module."""

import pytest
from datetime import datetime, timezone

from llm_benchmark.models import (
    BenchmarkResult,
    ConcurrentBatchResult,
    SweepConfigResult,
    SweepModelResult,
)
from llm_benchmark.config import DEFAULT_CONCURRENT, SWEEP_NUM_CTX, SWEEP_PROMPT


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


class TestAggregrateThroughput:
    """Placeholder for aggregate throughput tests -- filled in Task 2."""

    def test_placeholder(self):
        """Aggregate throughput = sum(tokens) / wall_time (tested in Task 2)."""
        # Will be replaced by actual test in Task 2
        assert True
