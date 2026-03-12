"""Pydantic data models for Ollama responses and benchmark results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator


class Message(BaseModel):
    """A chat message with role and content."""

    role: str
    content: str


class OllamaResponse(BaseModel):
    """Validated Ollama API response with timing and token metrics.

    Unlike the original benchmark.py validator, this does NOT print warnings
    as a side effect. Instead, it silently normalizes prompt_eval_count=-1
    to 0 and sets the prompt_cached flag.
    """

    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = -1
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int
    prompt_cached: bool = False

    @model_validator(mode="before")
    @classmethod
    def handle_prompt_caching(cls, data: dict) -> dict:
        """Normalize prompt_eval_count=-1 (caching) to 0 and flag it."""
        if isinstance(data, dict):
            if data.get("prompt_eval_count", -1) == -1:
                data["prompt_eval_count"] = 0
                data["prompt_cached"] = True
        return data


class BenchmarkResult(BaseModel):
    """Result of a single benchmark run (one prompt against one model)."""

    model: str
    prompt: str
    success: bool
    response: OllamaResponse | None = None
    error: str | None = None
    prompt_cached: bool = False


class ModelSummary(BaseModel):
    """Aggregated benchmark results for a single model across all prompts."""

    model: str
    results: list[BenchmarkResult]
    avg_prompt_eval_ts: float
    avg_response_ts: float
    avg_total_ts: float


class SystemInfo(BaseModel):
    """System hardware and software information."""

    cpu: str
    ram_gb: float
    gpu: str
    gpu_vram_gb: float | None = None
    os_name: str
    python_version: str
    ollama_version: str


def _ns_to_sec(ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return ns / 1_000_000_000


def compute_averages(results: list[BenchmarkResult]) -> dict:
    """Correct averaging: total_tokens / total_time, not mean of rates.

    This implements the STAB-04 fix: sum all tokens, sum all durations,
    divide once. The arithmetic mean of rates is mathematically incorrect.

    Args:
        results: List of benchmark results (may include failures).

    Returns:
        Dict with prompt_eval_ts, response_ts, total_ts keys,
        or empty dict if no successful results.
    """
    successful = [r for r in results if r.success and r.response is not None]
    if not successful:
        return {}

    total_prompt_tokens = sum(r.response.prompt_eval_count for r in successful)
    total_response_tokens = sum(r.response.eval_count for r in successful)
    total_prompt_time = sum(
        _ns_to_sec(r.response.prompt_eval_duration) for r in successful
    )
    total_response_time = sum(
        _ns_to_sec(r.response.eval_duration) for r in successful
    )

    total_time = total_prompt_time + total_response_time
    total_tokens = total_prompt_tokens + total_response_tokens

    return {
        "prompt_eval_ts": (
            total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0
        ),
        "response_ts": (
            total_response_tokens / total_response_time
            if total_response_time > 0
            else 0
        ),
        "total_ts": total_tokens / total_time if total_time > 0 else 0,
    }
