"""LM Studio backend implementation.

Wraps the LM Studio HTTP API to satisfy the Backend protocol. Derives
eval_duration from tokens_per_second in the stats object and maps httpx
exceptions to BackendError.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Iterator

import httpx

from llm_benchmark.backends import BackendError, BackendResponse, StreamResult


class LMStudioBackend:
    """Backend implementation for LM Studio."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1234) -> None:
        self._base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=self._base_url, timeout=600.0)

    @property
    def name(self) -> str:
        """Return backend identifier."""
        return "lm-studio"

    @property
    def version(self) -> str:
        """Get LM Studio version via lms CLI."""
        try:
            result = subprocess.run(
                ["lms", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
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
        """Send a chat request to LM Studio.

        Args:
            model: Model name/identifier.
            messages: List of message dicts with 'role' and 'content'.
            stream: If True, return StreamResult for incremental output.
            options: Optional dict of model parameters.

        Returns:
            BackendResponse for non-streaming, StreamResult for streaming.

        Raises:
            BackendError: On any HTTP communication error.
        """
        try:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": stream,
            }
            if options:
                payload.update(options)

            if stream:
                return self._handle_stream(model, payload)

            resp = self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return self._to_response(data)

        except httpx.ConnectError as e:
            raise BackendError(str(e), retryable=True, original=e) from e
        except httpx.TimeoutException as e:
            raise BackendError(str(e), retryable=True, original=e) from e
        except httpx.HTTPStatusError as e:
            retryable = e.response.status_code >= 500
            raise BackendError(str(e), retryable=retryable, original=e) from e

    def _handle_stream(
        self, model: str, payload: dict[str, Any]
    ) -> StreamResult:
        """Handle streaming chat response via SSE."""
        collected_content: list[str] = []
        final_data: dict[str, Any] = {}

        stream_ctx = self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        )
        raw_resp = stream_ctx.__enter__()

        def chunk_generator() -> Iterator[str]:
            nonlocal final_data
            try:
                for line in raw_resp.iter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract content from delta
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected_content.append(content)
                            yield content

                    # Capture usage/stats from final chunk
                    if chunk.get("usage") or chunk.get("stats"):
                        final_data = chunk
            finally:
                stream_ctx.__exit__(None, None, None)

        def finalize() -> BackendResponse:
            content = "".join(collected_content)
            usage = final_data.get("usage", {})
            stats = final_data.get("stats", {})

            eval_count = usage.get("completion_tokens", 0)
            tps = stats.get("tokens_per_second", 0)
            eval_duration = eval_count / tps if tps > 0 else 0.0

            return BackendResponse(
                model=final_data.get("model", model),
                content=content,
                done=True,
                prompt_eval_count=usage.get("prompt_tokens", 0),
                eval_count=eval_count,
                eval_duration=eval_duration,
                total_duration=eval_duration,
                prompt_eval_duration=0.0,
                load_duration=0.0,
            )

        return StreamResult(chunks=chunk_generator(), finalize=finalize)

    def _to_response(self, data: dict[str, Any]) -> BackendResponse:
        """Convert LM Studio chat response to BackendResponse.

        Derives eval_duration from eval_count / tokens_per_second.
        """
        choices = data.get("choices", [])
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        usage = data.get("usage", {})
        stats = data.get("stats", {})

        eval_count = usage.get("completion_tokens", 0)
        prompt_eval_count = usage.get("prompt_tokens", 0)
        tps = stats.get("tokens_per_second", 0)

        eval_duration = eval_count / tps if tps > 0 else 0.0

        return BackendResponse(
            model=data.get("model", ""),
            content=content,
            done=True,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            eval_duration=eval_duration,
            total_duration=eval_duration,
            prompt_eval_duration=0.0,
            load_duration=0.0,
        )

    def list_models(self) -> list[dict[str, Any]]:
        """List all downloaded models from LM Studio.

        Queries /api/v1/models for the full catalog (not just loaded models).
        """
        try:
            resp = self._client.get("/api/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return [
                {"model": m.get("id", ""), "size": 0}
                for m in data.get("data", [])
            ]
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            raise BackendError(str(e), retryable=True, original=e) from e

    def unload_model(self, model: str) -> bool:
        """Unload a model from LM Studio via /api/v1/models/unload.

        Args:
            model: Model identifier to unload.

        Returns:
            True on success, False on error.
        """
        try:
            resp = self._client.post(
                "/api/v1/models/unload", json={"model": model}
            )
            resp.raise_for_status()
            return True
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            return False

    def warmup(self, model: str, timeout: float) -> bool:
        """Warm up by sending a short chat request.

        Args:
            model: Model name.
            timeout: Maximum seconds to wait (advisory).

        Returns:
            True if warmup succeeded, False otherwise.
        """
        try:
            self.chat(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except BackendError:
            return False

    def detect_context_window(self, model: str) -> int:
        """Return default context window size.

        LM Studio does not reliably expose context window information.

        Returns:
            Default 4096.
        """
        return 4096

    def get_model_size(self, model: str) -> float | None:
        """Return None -- model size not available from LM Studio API."""
        return None

    def check_connectivity(self) -> bool:
        """Check if LM Studio is reachable via /v1/models endpoint.

        Returns:
            True if server responds with 200, False otherwise.
        """
        try:
            resp = self._client.get("/v1/models", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
