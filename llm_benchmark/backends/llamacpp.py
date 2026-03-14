"""llama.cpp server backend implementation.

Wraps the llama-server HTTP API to satisfy the Backend protocol. Converts
millisecond timing data from the timings object to seconds and maps httpx
exceptions to BackendError.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Iterator

import httpx

from llm_benchmark.backends import BackendError, BackendResponse, StreamResult


def _ms_to_sec(ms: float) -> float:
    """Convert milliseconds to seconds."""
    return ms / 1000.0


class LlamaCppBackend:
    """Backend implementation for llama.cpp server (llama-server)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        self._base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=self._base_url, timeout=600.0)
        self._server_process: subprocess.Popen | None = None

    @property
    def name(self) -> str:
        """Return backend identifier."""
        return "llama-cpp"

    @property
    def version(self) -> str:
        """Get llama-server version from /props or CLI fallback."""
        try:
            resp = self._client.get("/props")
            if resp.status_code == 200:
                data = resp.json()
                build = data.get("build", {})
                if isinstance(build, dict):
                    build_num = build.get("number", "")
                    if build_num:
                        return str(build_num)
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            pass
        try:
            result = subprocess.run(
                ["llama-server", "--version"],
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
        """Send a chat request to llama-server.

        Args:
            model: Model name (informational -- llama-server serves one model).
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

        # Open streaming connection
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

                    # Capture timing data from final chunk
                    if chunk.get("timings") or (
                        choices
                        and choices[0].get("finish_reason") == "stop"
                    ):
                        final_data = chunk
            finally:
                stream_ctx.__exit__(None, None, None)

        def finalize() -> BackendResponse:
            content = "".join(collected_content)
            timings = final_data.get("timings", {})
            return BackendResponse(
                model=final_data.get("model", model),
                content=content,
                done=True,
                prompt_eval_count=timings.get("prompt_n", 0),
                eval_count=timings.get("predicted_n", 0),
                prompt_eval_duration=_ms_to_sec(timings.get("prompt_ms", 0)),
                eval_duration=_ms_to_sec(timings.get("predicted_ms", 0)),
                total_duration=_ms_to_sec(
                    timings.get("prompt_ms", 0)
                    + timings.get("predicted_ms", 0)
                ),
                load_duration=0.0,
            )

        return StreamResult(chunks=chunk_generator(), finalize=finalize)

    def _to_response(self, data: dict[str, Any]) -> BackendResponse:
        """Convert llama.cpp chat response to BackendResponse.

        Timing fields in the timings object are in milliseconds.
        """
        choices = data.get("choices", [])
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        timings = data.get("timings", {})

        prompt_ms = timings.get("prompt_ms", 0)
        predicted_ms = timings.get("predicted_ms", 0)

        return BackendResponse(
            model=data.get("model", ""),
            content=content,
            done=True,
            prompt_eval_count=timings.get("prompt_n", 0),
            eval_count=timings.get("predicted_n", 0),
            prompt_eval_duration=_ms_to_sec(prompt_ms),
            eval_duration=_ms_to_sec(predicted_ms),
            total_duration=_ms_to_sec(prompt_ms + predicted_ms),
            load_duration=0.0,
        )

    def list_models(self) -> list[dict[str, Any]]:
        """List the model loaded in llama-server.

        llama-server serves one model at a time. Returns a list with a
        single entry from the /v1/models endpoint.
        """
        try:
            resp = self._client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("data", []):
                models.append({"model": m.get("id", ""), "size": 0})
            return models
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            raise BackendError(str(e), retryable=True, original=e) from e

    def unload_model(self, model: str) -> bool:
        """Terminate managed server process if one exists.

        For llama-server, unloading means stopping the server since it
        serves a single model.
        """
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None
        return True

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
        """Detect context window size from /props endpoint.

        Returns:
            Context window size in tokens (default 4096 if undetectable).
        """
        try:
            resp = self._client.get("/props")
            resp.raise_for_status()
            data = resp.json()
            gen_settings = data.get("default_generation_settings", {})
            n_ctx = gen_settings.get("n_ctx", 0)
            if n_ctx > 0:
                return n_ctx
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            pass
        return 4096

    def get_model_size(self, model: str) -> float | None:
        """Return None -- model size not available from llama-server API."""
        return None

    def check_connectivity(self) -> bool:
        """Check if llama-server is reachable via /health endpoint.

        Returns:
            True if server responds with 200, False otherwise.
        """
        try:
            resp = self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
