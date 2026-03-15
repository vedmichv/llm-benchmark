"""Pydantic data models for benchmark results."""

from __future__ import annotations

from pydantic import BaseModel

from llm_benchmark.backends import BackendResponse


class Message(BaseModel):
    """A chat message with role and content."""

    role: str
    content: str


class BenchmarkResult(BaseModel):
    """Result of a single benchmark run (one prompt against one model)."""

    model: str
    prompt: str
    success: bool
    response: BackendResponse | None = None
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
    backend_name: str
    backend_version: str


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
