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


class ConcurrentBatchResult(BaseModel):
    """Result of a concurrent batch of N parallel requests to a single model."""

    model: str
    prompt: str
    num_workers: int
    wall_time_s: float
    results: list[BenchmarkResult]
    aggregate_throughput_ts: float
    avg_request_throughput_ts: float


class SweepConfigResult(BaseModel):
    """Result of a single configuration sweep (one num_ctx / num_gpu combo)."""

    model: str
    num_ctx: int
    num_gpu: int
    response_ts: float
    total_ts: float
    eval_count: int
    total_duration_s: float
    success: bool
    error: str | None = None


class SweepModelResult(BaseModel):
    """Aggregated sweep results for one model across all configurations."""

    model: str
    configs: list[SweepConfigResult]
    best_config: SweepConfigResult | None = None


def _ns_to_sec(ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return ns / 1_000_000_000


