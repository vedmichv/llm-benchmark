"""Ollama backend implementation.

Wraps the ollama Python SDK to satisfy the Backend protocol. Converts
nanosecond timing data to seconds and maps ollama exceptions to BackendError.
"""

from __future__ import annotations

import subprocess
from typing import Any, Iterator

import ollama
from ollama import RequestError as _OllamaRequestError
from ollama import ResponseError as _OllamaResponseError

from llm_benchmark.backends import BackendError, BackendResponse, StreamResult


def _ns_to_sec(ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return ns / 1_000_000_000


class OllamaBackend:
    """Backend implementation for Ollama."""

    @property
    def name(self) -> str:
        """Return backend identifier."""
        return "ollama"

    @property
    def version(self) -> str:
        """Get installed Ollama version via CLI."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                version = version.replace("ollama version is ", "")
                version = version.replace("ollama version ", "")
                return version
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return "Unknown"

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> BackendResponse | StreamResult:
        """Send a chat request to Ollama.

        Args:
            model: Model name (e.g. "llama3").
            messages: List of message dicts with 'role' and 'content'.
            stream: If True, return StreamResult for incremental output.
            options: Optional dict of model parameters (num_ctx, num_gpu, etc.).

        Returns:
            BackendResponse for non-streaming, StreamResult for streaming.

        Raises:
            BackendError: On any Ollama communication error.
        """
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if stream:
                kwargs["stream"] = True
            if options is not None:
                kwargs["options"] = options

            raw = ollama.chat(**kwargs)

            if stream:
                return self._handle_stream(raw, model)

            # Non-streaming: convert response
            data = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
            return self._to_response(data)

        except _OllamaRequestError as e:
            raise BackendError(str(e), retryable=True, original=e) from e
        except _OllamaResponseError as e:
            status = getattr(e, "status_code", 0)
            retryable = status >= 500
            raise BackendError(str(e), retryable=retryable, original=e) from e
        except ConnectionError as e:
            raise BackendError(str(e), retryable=True, original=e) from e

    def _handle_stream(
        self, stream_iter: Any, model: str
    ) -> StreamResult:
        """Wrap an Ollama streaming response into a StreamResult."""
        collected_content: list[str] = []
        final_data: dict[str, Any] = {}

        def chunk_generator() -> Iterator[str]:
            nonlocal final_data
            for chunk in stream_iter:
                # Extract chunk data
                chunk_dict = (
                    chunk if isinstance(chunk, dict) else
                    chunk.model_dump() if hasattr(chunk, "model_dump") else
                    dict(chunk)
                )

                # Yield content
                msg = chunk_dict.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if content:
                    collected_content.append(content)
                    yield content

                # Capture final chunk with timing data
                done = chunk_dict.get("done", False)
                if done:
                    final_data = chunk_dict

        def finalize() -> BackendResponse:
            # Build response from final chunk timing data
            data = dict(final_data)
            # Ensure content is the full accumulated text
            if "message" not in data or not data["message"]:
                data["message"] = {"role": "assistant", "content": ""}
            content = "".join(collected_content)
            if isinstance(data.get("message"), dict):
                data["message"]["content"] = content
            # Set model if missing
            if "model" not in data:
                data["model"] = model
            return self._to_response(data)

        chunks = chunk_generator()
        return StreamResult(chunks=chunks, finalize=finalize)

    def _to_response(self, raw: dict[str, Any]) -> BackendResponse:
        """Convert raw Ollama response dict (nanoseconds) to BackendResponse (seconds).

        Detects prompt_eval_count == -1 as prompt caching, normalizing to 0.
        """
        msg = raw.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")

        prompt_eval_count = raw.get("prompt_eval_count", 0)
        prompt_cached = prompt_eval_count == -1
        if prompt_cached:
            prompt_eval_count = 0

        return BackendResponse(
            model=raw.get("model", ""),
            content=content,
            done=raw.get("done", True),
            prompt_eval_count=prompt_eval_count,
            eval_count=raw.get("eval_count", 0),
            total_duration=_ns_to_sec(raw.get("total_duration", 0)),
            load_duration=_ns_to_sec(raw.get("load_duration", 0)),
            prompt_eval_duration=_ns_to_sec(raw.get("prompt_eval_duration", 0)),
            eval_duration=_ns_to_sec(raw.get("eval_duration", 0)),
            prompt_cached=prompt_cached,
        )

    def list_models(self) -> list[dict[str, Any]]:
        """List available models from Ollama.

        Returns:
            List of dicts with at minimum 'model' and 'size' keys.
        """
        try:
            response = ollama.list()
            models = []
            for m in response.models:
                model_dict = m.model_dump() if hasattr(m, "model_dump") else dict(m)
                # Ensure 'model' key exists (ollama SDK uses 'name' in some versions)
                if "model" not in model_dict and "name" in model_dict:
                    model_dict["model"] = model_dict["name"]
                models.append(model_dict)
            return models
        except (_OllamaRequestError, _OllamaResponseError, ConnectionError) as e:
            raise BackendError(str(e), retryable=True, original=e) from e

    def unload_model(self, model: str) -> bool:
        """Unload a model from Ollama's memory.

        Uses keep_alive=0 to signal immediate unload.
        """
        try:
            ollama.generate(model=model, prompt="", keep_alive=0)
            return True
        except (_OllamaRequestError, _OllamaResponseError, ConnectionError):
            return False

    def warmup(self, model: str, timeout: float) -> bool:
        """Warm up a model by sending a short prompt.

        Args:
            model: Model name.
            timeout: Maximum seconds to wait (advisory, not enforced here).

        Returns:
            True if warmup succeeded, False otherwise.
        """
        try:
            ollama.chat(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except (_OllamaRequestError, _OllamaResponseError, ConnectionError):
            return False

    def detect_context_window(self, model: str) -> int:
        """Detect the context window size for a model.

        Returns:
            Context window size in tokens (default 4096 if undetectable).
        """
        try:
            info = ollama.show(model)
            model_info = getattr(info, "modelinfo", None) or {}
            if not isinstance(model_info, dict):
                model_info = (
                    model_info.model_dump()
                    if hasattr(model_info, "model_dump")
                    else dict(model_info)
                )
            # Look for context_length in various keys
            for key in model_info:
                if "context_length" in key:
                    return int(model_info[key])
            return 4096
        except (_OllamaRequestError, _OllamaResponseError, ConnectionError):
            return 4096

    def get_model_size(self, model: str) -> float | None:
        """Get model size in GB.

        Returns:
            Size in GB, or None if model not found.
        """
        try:
            response = ollama.list()
            for m in response.models:
                m_dict = m.model_dump() if hasattr(m, "model_dump") else dict(m)
                name = m_dict.get("model", m_dict.get("name", ""))
                if name == model:
                    size_bytes = m_dict.get("size", 0)
                    return size_bytes / (1024**3) if size_bytes else None
            return None
        except (_OllamaRequestError, _OllamaResponseError, ConnectionError):
            return None

    def check_connectivity(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if server responds, False otherwise.
        """
        try:
            ollama.list()
            return True
        except (_OllamaRequestError, _OllamaResponseError, ConnectionError):
            return False
