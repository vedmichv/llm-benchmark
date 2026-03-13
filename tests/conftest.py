"""Shared test fixtures."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from llm_benchmark.models import BenchmarkResult, OllamaResponse


@pytest.fixture
def sample_ollama_response_dict():
    """Return a valid Ollama response dict with realistic nanosecond values."""
    return {
        "model": "llama3.2:1b",
        "created_at": datetime.now(timezone.utc).isoformat(),
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
def sample_benchmark_result(sample_ollama_response_dict):
    """Return a successful BenchmarkResult with a valid OllamaResponse."""
    response = OllamaResponse.model_validate(sample_ollama_response_dict)
    return BenchmarkResult(
        model="llama3.2:1b",
        prompt="Why is the sky blue?",
        success=True,
        response=response,
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
