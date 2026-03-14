"""Shared test fixtures."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_benchmark.backends import BackendResponse
from llm_benchmark.models import BenchmarkResult


@pytest.fixture
def sample_ollama_response_dict():
    """Return a valid Ollama response dict with realistic nanosecond values.

    Note: This fixture retains nanosecond values for backward compatibility
    with tests that test the OllamaBackend conversion layer. Tests for
    modules that use BackendResponse directly should use sample_backend_response.
    """
    return {
        "model": "llama3.2:1b",
        "created_at": datetime.now(UTC).isoformat(),
        "message": {"role": "assistant", "content": "The sky is blue because..."},
        "done": True,
        "total_duration": 5_000_000_000,  # 5 seconds
        "load_duration": 500_000_000,  # 0.5 seconds
        "prompt_eval_count": 15,
        "prompt_eval_duration": 200_000_000,  # 0.2 seconds
        "eval_count": 120,
        "eval_duration": 4_000_000_000,  # 4 seconds
    }


@pytest.fixture
def cached_ollama_response_dict(sample_ollama_response_dict):
    """Return an Ollama response dict simulating prompt caching."""
    data = sample_ollama_response_dict.copy()
    data["prompt_eval_count"] = -1
    return data


@pytest.fixture
def sample_backend_response():
    """Return a BackendResponse with realistic seconds-based values."""
    return BackendResponse(
        model="llama3.2:1b",
        content="The sky is blue because...",
        done=True,
        prompt_eval_count=15,
        eval_count=120,
        total_duration=5.0,
        load_duration=0.5,
        prompt_eval_duration=0.2,
        eval_duration=4.0,
        prompt_cached=False,
    )


@pytest.fixture
def sample_benchmark_result(sample_backend_response):
    """Return a successful BenchmarkResult with a valid BackendResponse."""
    return BenchmarkResult(
        model="llama3.2:1b",
        prompt="Why is the sky blue?",
        success=True,
        response=sample_backend_response,
    )


@pytest.fixture
def mock_async_client(sample_ollama_response_dict):
    """Provide a mock ollama.AsyncClient with configurable chat() return value.

    The mock's chat() returns a MagicMock with model_dump() returning the
    sample response dict, matching ollama SDK ChatResponse behavior.
    """
    client = AsyncMock()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = sample_ollama_response_dict
    client.chat = AsyncMock(return_value=mock_response)
    return client
