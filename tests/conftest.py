"""Shared test fixtures."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
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
def cached_backend_response():
    """Return a BackendResponse simulating prompt caching."""
    return BackendResponse(
        model="llama3.2:1b",
        content="The sky is blue because...",
        done=True,
        total_duration=5.0,
        load_duration=0.5,
        prompt_eval_count=0,
        prompt_eval_duration=0.0,
        eval_count=120,
        eval_duration=4.0,
        prompt_cached=True,
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
def mock_backend(sample_backend_response):
    """Provide a mock Backend with default behavior for all protocol methods."""
    backend = MagicMock()
    backend.name = "ollama"
    backend.version = "0.6.1"
    backend.chat.return_value = sample_backend_response
    backend.list_models.return_value = [{"model": "llama3.2:1b", "size": 1_073_741_824}]
    backend.check_connectivity.return_value = True
    backend.warmup.return_value = True
    backend.unload_model.return_value = True
    backend.detect_context_window.return_value = 4096
    backend.get_model_size.return_value = 1.0
    return backend


# ---------------------------------------------------------------------------
# httpx mock response factory
# ---------------------------------------------------------------------------


def make_httpx_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
    headers: dict | None = None,
) -> httpx.Response:
    """Create a mock httpx.Response with the given data."""
    resp = httpx.Response(
        status_code=status_code,
        headers=headers or {},
        json=json_data,
        text=text if json_data is None else "",
    )
    return resp


@pytest.fixture
def mock_httpx_response():
    """Factory fixture that creates mock httpx.Response objects."""
    return make_httpx_response


@pytest.fixture
def llamacpp_chat_response():
    """Sample llama.cpp /v1/chat/completions JSON with timings object."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The sky is blue because of Rayleigh scattering.",
                },
                "finish_reason": "stop",
            }
        ],
        "model": "my-model",
        "timings": {
            "prompt_n": 10,
            "prompt_ms": 30.0,
            "prompt_per_token_ms": 3.0,
            "prompt_per_second": 333.33,
            "predicted_n": 35,
            "predicted_ms": 660.0,
            "predicted_per_token_ms": 18.857,
            "predicted_per_second": 53.03,
        },
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 35,
            "total_tokens": 45,
        },
    }


@pytest.fixture
def lmstudio_chat_response():
    """Sample LM Studio /v1/chat/completions JSON with usage + stats."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The sky appears blue due to scattering.",
                },
                "finish_reason": "stop",
            }
        ],
        "model": "my-lmstudio-model",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 35,
            "total_tokens": 45,
        },
        "stats": {
            "tokens_per_second": 45.2,
        },
    }
