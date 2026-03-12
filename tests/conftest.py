"""Shared test fixtures."""

import pytest
from datetime import datetime, timezone


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
