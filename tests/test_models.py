"""Tests for Pydantic data models."""

import io
import sys

import pytest


def test_ollama_response_valid(sample_ollama_response_dict):
    """OllamaResponse validates a complete response dict successfully."""
    from llm_benchmark.models import OllamaResponse

    resp = OllamaResponse.model_validate(sample_ollama_response_dict)
    assert resp.model == "llama3.2:1b"
    assert resp.eval_count == 120
    assert resp.prompt_eval_count == 15
    assert resp.prompt_cached is False


def test_ollama_response_prompt_cached(cached_ollama_response_dict):
    """prompt_eval_count=-1 sets to 0, sets prompt_cached=True, no print side effect."""
    from llm_benchmark.models import OllamaResponse

    # Capture stdout to verify no print side effect
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    resp = OllamaResponse.model_validate(cached_ollama_response_dict)

    sys.stdout = old_stdout
    assert resp.prompt_eval_count == 0
    assert resp.prompt_cached is True
    assert captured.getvalue() == "", "Validator should not print to stdout"


def test_benchmark_result_success(sample_ollama_response_dict):
    """BenchmarkResult with success=True has response data."""
    from llm_benchmark.models import BenchmarkResult, OllamaResponse

    ollama_resp = OllamaResponse.model_validate(sample_ollama_response_dict)
    result = BenchmarkResult(
        model="llama3.2:1b",
        prompt="Why is the sky blue?",
        success=True,
        response=ollama_resp,
    )
    assert result.success is True
    assert result.response is not None
    assert result.error is None
    assert result.prompt_cached is False


def test_benchmark_result_failure():
    """BenchmarkResult with success=False has error message."""
    from llm_benchmark.models import BenchmarkResult

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
    from llm_benchmark.models import Message

    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_compute_averages_correct(sample_ollama_response_dict):
    """compute_averages uses total_tokens/total_time (STAB-04)."""
    from llm_benchmark.models import BenchmarkResult, OllamaResponse
    from llm_benchmark.runner import compute_averages

    # Create two results with known values
    resp1 = OllamaResponse.model_validate(sample_ollama_response_dict)
    resp2 = OllamaResponse.model_validate(sample_ollama_response_dict)

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
