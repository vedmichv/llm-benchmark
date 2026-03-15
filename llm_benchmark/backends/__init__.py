"""Backend abstraction layer for LLM benchmark.

Provides the Backend Protocol, BackendResponse model, BackendError exception,
StreamResult wrapper, and create_backend() factory function.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class Backend(Protocol):
    """Protocol defining the interface all LLM backends must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> BackendResponse | StreamResult: ...

    def list_models(self) -> list[dict[str, Any]]: ...

    def unload_model(self, model: str) -> bool: ...

    def warmup(self, model: str, timeout: float) -> bool: ...

    def detect_context_window(self, model: str) -> int: ...

    def get_model_size(self, model: str) -> float | None: ...

    def check_connectivity(self) -> bool: ...


class BackendResponse(BaseModel):
    """Normalized response from any LLM backend.

    All timing fields are in seconds (float), not nanoseconds.
    Each backend converts its native timing format internally.
    """

    model: str
    content: str
    done: bool
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration: float = 0.0
    load_duration: float = 0.0
    prompt_eval_duration: float = 0.0
    eval_duration: float = 0.0
    prompt_cached: bool = False


class BackendError(Exception):
    """Unified exception for all backend errors.

    Attributes:
        retryable: Whether the operation can be retried.
        original: The original exception, if any.
    """

    def __init__(
        self,
        message: str,
        retryable: bool = False,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.original = original


class StreamResult:
    """Wrapper for streaming chat responses.

    Provides a chunks iterator for real-time content display and a
    deferred response property for timing data after iteration completes.
    """

    def __init__(
        self,
        chunks: Iterator[str],
        finalize: Any,  # callable returning BackendResponse
    ) -> None:
        self._chunks = chunks
        self._finalize = finalize
        self._response: BackendResponse | None = None

    @property
    def chunks(self) -> Iterator[str]:
        """Iterator yielding content strings as they arrive."""
        return self._chunks

    @property
    def response(self) -> BackendResponse:
        """BackendResponse with timing data, available after iteration."""
        if self._response is None:
            self._response = self._finalize()
        return self._response


def create_backend(
    name: str = "ollama",
    *,
    host: str | None = None,
    port: int | None = None,
) -> Backend:
    """Factory function to create a backend instance by name.

    Args:
        name: Backend identifier ('ollama', 'llama-cpp', 'lm-studio').
        host: Optional hostname override for the backend server.
        port: Optional port override for the backend server.

    Returns:
        A Backend instance.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    if name == "ollama":
        from llm_benchmark.backends.ollama import OllamaBackend

        return OllamaBackend()

    if name == "llama-cpp":
        from llm_benchmark.backends.llamacpp import LlamaCppBackend

        kwargs: dict[str, Any] = {}
        if host is not None:
            kwargs["host"] = host
        if port is not None:
            kwargs["port"] = port
        return LlamaCppBackend(**kwargs)

    if name == "lm-studio":
        from llm_benchmark.backends.lmstudio import LMStudioBackend

        kwargs_lms: dict[str, Any] = {}
        if host is not None:
            kwargs_lms["host"] = host
        if port is not None:
            kwargs_lms["port"] = port
        return LMStudioBackend(**kwargs_lms)

    raise ValueError(
        f"Unknown backend: {name!r}. Supported backends: 'ollama', 'llama-cpp', 'lm-studio'"
    )
