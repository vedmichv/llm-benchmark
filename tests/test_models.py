"""Tests for Pydantic data models."""

import pytest

from llm_benchmark.backends import BackendResponse
from llm_benchmark.models import BenchmarkResult, Message, SystemInfo


def test_backend_response_fields():
    """BackendResponse stores timing in seconds."""
    resp = BackendResponse(
        model="llama3.2:1b",
        content="The sky is blue because...",
        done=True,
        prompt_eval_count=15,
        eval_count=120,
        total_duration=5.0,
        load_duration=0.5,
        prompt_eval_duration=0.2,
        eval_duration=4.0,
    )
    assert resp.model == "llama3.2:1b"
    assert resp.eval_count == 120
    assert resp.prompt_eval_count == 15
    assert resp.prompt_cached is False
    assert resp.eval_duration == 4.0


def test_benchmark_result_success(sample_backend_response):
    """BenchmarkResult with success=True has response data."""
    result = BenchmarkResult(
        model="llama3.2:1b",
        prompt="Why is the sky blue?",
        success=True,
        response=sample_backend_response,
    )
    assert result.success is True
    assert result.response is not None
    assert result.error is None
    assert result.prompt_cached is False


def test_benchmark_result_failure():
    """BenchmarkResult with success=False has error message."""
    result = BenchmarkResult(
        model="llama3.2:1b",
        prompt="Why is the sky blue?",
        success=False,
        error="Connection timed out",
    )
    assert result.success is False
    assert result.response is None
    assert result.error == "Connection timed out"


def test_message_model():
    """Message model has role and content fields."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_system_info_backend_fields():
    """SystemInfo has backend_name and backend_version fields."""
    info = SystemInfo(
        cpu="Apple M2",
        ram_gb=16.0,
        gpu="Apple M2 (integrated)",
        os_name="macOS 14.0",
        python_version="3.12.0",
        backend_name="ollama",
        backend_version="0.6.1",
    )
    assert info.backend_name == "ollama"
    assert info.backend_version == "0.6.1"


def test_compute_averages_correct():
    """compute_averages uses total_tokens/total_time (STAB-04)."""
    from llm_benchmark.runner import compute_averages

    # Create two results with known values (seconds-based)
    resp1 = BackendResponse(
        model="test", content="", done=True,
        prompt_eval_count=15, eval_count=120,
        total_duration=5.0, prompt_eval_duration=0.2, eval_duration=4.0,
    )
    resp2 = BackendResponse(
        model="test", content="", done=True,
        prompt_eval_count=15, eval_count=120,
        total_duration=5.0, prompt_eval_duration=0.2, eval_duration=4.0,
    )

    results = [
        BenchmarkResult(model="test", prompt="p1", success=True, response=resp1),
        BenchmarkResult(model="test", prompt="p2", success=True, response=resp2),
    ]

    averages = compute_averages(results)

    # Each response: prompt_eval_count=15, prompt_eval_duration=0.2s
    # Total prompt tokens = 30, total prompt time = 0.4s => 75 t/s
    assert averages["prompt_eval_ts"] == pytest.approx(75.0)

    # Each response: eval_count=120, eval_duration=4.0s
    # Total response tokens = 240, total response time = 8.0s => 30 t/s
    assert averages["response_ts"] == pytest.approx(30.0)

    # Total tokens = 270, total time = 8.4s => ~32.14 t/s
    assert averages["total_ts"] == pytest.approx(270.0 / 8.4)


def test_compute_averages_empty():
    """compute_averages returns empty dict for no successful results."""
    from llm_benchmark.runner import compute_averages

    assert compute_averages([]) == {}
