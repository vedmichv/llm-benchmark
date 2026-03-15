"""Tests for the LlamaCppBackend implementation."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_benchmark.backends import Backend, BackendError, BackendResponse, StreamResult
from llm_benchmark.backends.llamacpp import LlamaCppBackend

# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """LlamaCppBackend satisfies the Backend Protocol."""

    def test_isinstance_backend(self):
        backend = LlamaCppBackend()
        assert isinstance(backend, Backend)

    def test_name_property(self):
        backend = LlamaCppBackend()
        assert backend.name == "llama-cpp"

    def test_version_property_returns_string(self):
        backend = LlamaCppBackend()
        assert isinstance(backend.version, str)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Constructor accepts host and port with defaults."""

    def test_default_host_port(self):
        backend = LlamaCppBackend()
        assert "127.0.0.1" in backend._base_url
        assert "8080" in backend._base_url

    def test_custom_host_port(self):
        backend = LlamaCppBackend(host="192.168.1.1", port=9090)
        assert "192.168.1.1" in backend._base_url
        assert "9090" in backend._base_url


# ---------------------------------------------------------------------------
# chat() non-streaming
# ---------------------------------------------------------------------------


class TestChatNonStreaming:
    """chat() sends POST and returns BackendResponse with timing conversion."""

    def test_chat_returns_backend_response(self, llamacpp_chat_response, mock_httpx_response):
        backend = LlamaCppBackend()
        mock_resp = mock_httpx_response(json_data=llamacpp_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("my-model", [{"role": "user", "content": "Hi"}])
        assert isinstance(result, BackendResponse)

    def test_chat_timing_conversion_ms_to_seconds(self, llamacpp_chat_response, mock_httpx_response):
        """Timings from ms are converted to seconds."""
        backend = LlamaCppBackend()
        mock_resp = mock_httpx_response(json_data=llamacpp_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("my-model", [{"role": "user", "content": "Hi"}])
        assert result.prompt_eval_duration == pytest.approx(0.03)
        assert result.eval_duration == pytest.approx(0.66)
        assert result.prompt_eval_count == 10
        assert result.eval_count == 35

    def test_chat_content_extracted(self, llamacpp_chat_response, mock_httpx_response):
        backend = LlamaCppBackend()
        mock_resp = mock_httpx_response(json_data=llamacpp_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("my-model", [{"role": "user", "content": "Hi"}])
        assert "Rayleigh scattering" in result.content

    def test_chat_model_field(self, llamacpp_chat_response, mock_httpx_response):
        backend = LlamaCppBackend()
        mock_resp = mock_httpx_response(json_data=llamacpp_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("my-model", [{"role": "user", "content": "Hi"}])
        assert result.model == "my-model"


# ---------------------------------------------------------------------------
# chat() streaming
# ---------------------------------------------------------------------------


class TestChatStreaming:
    """chat(stream=True) returns StreamResult with timing data."""

    def _make_sse_lines(self):
        """Create SSE lines simulating llama.cpp streaming response."""
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"timings":{"prompt_n":5,"prompt_ms":20.0,"predicted_n":2,"predicted_ms":100.0},"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
            "data: [DONE]\n\n",
        ]
        return chunks

    def test_stream_returns_stream_result(self):
        backend = LlamaCppBackend()
        lines = self._make_sse_lines()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(
            line.strip() for line in lines if line.strip()
        )
        mock_resp.headers = {"content-type": "text/event-stream"}

        with patch.object(backend._client, "stream") as mock_stream:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_resp)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_cm

            result = backend.chat(
                "my-model",
                [{"role": "user", "content": "Hi"}],
                stream=True,
            )
        assert isinstance(result, StreamResult)

    def test_stream_chunks_yield_content(self):
        backend = LlamaCppBackend()
        lines = self._make_sse_lines()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(
            line.strip() for line in lines if line.strip()
        )

        with patch.object(backend._client, "stream") as mock_stream:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_resp)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_cm

            result = backend.chat(
                "my-model",
                [{"role": "user", "content": "Hi"}],
                stream=True,
            )
            content_parts = list(result.chunks)
        assert "Hello" in content_parts
        assert " world" in content_parts

    def test_stream_response_has_timings(self):
        backend = LlamaCppBackend()
        lines = self._make_sse_lines()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(
            line.strip() for line in lines if line.strip()
        )

        with patch.object(backend._client, "stream") as mock_stream:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_resp)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_cm

            result = backend.chat(
                "my-model",
                [{"role": "user", "content": "Hi"}],
                stream=True,
            )
            # Consume chunks to populate timing data
            list(result.chunks)
            resp = result.response
        assert resp.prompt_eval_count == 5
        assert resp.eval_count == 2
        assert resp.prompt_eval_duration == pytest.approx(0.02)
        assert resp.eval_duration == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    """list_models() queries /v1/models endpoint."""

    def test_list_models_returns_list(self, mock_httpx_response):
        backend = LlamaCppBackend()
        resp_data = {
            "data": [{"id": "my-model", "object": "model"}]
        }
        mock_resp = mock_httpx_response(json_data=resp_data)
        with patch.object(backend._client, "get", return_value=mock_resp):
            models = backend.list_models()
        assert len(models) == 1
        assert models[0]["model"] == "my-model"


# ---------------------------------------------------------------------------
# check_connectivity
# ---------------------------------------------------------------------------


class TestCheckConnectivity:
    """check_connectivity() checks /health endpoint."""

    def test_connectivity_true_on_200(self, mock_httpx_response):
        backend = LlamaCppBackend()
        mock_resp = mock_httpx_response(
            json_data={"status": "ok"}
        )
        with patch.object(backend._client, "get", return_value=mock_resp):
            assert backend.check_connectivity() is True

    def test_connectivity_false_on_error(self):
        backend = LlamaCppBackend()
        with patch.object(
            backend._client, "get", side_effect=httpx.ConnectError("refused")
        ):
            assert backend.check_connectivity() is False


# ---------------------------------------------------------------------------
# unload_model
# ---------------------------------------------------------------------------


class TestUnloadModel:
    """unload_model() terminates managed server process."""

    def test_unload_with_process(self):
        backend = LlamaCppBackend()
        mock_proc = MagicMock()
        backend._server_process = mock_proc
        result = backend.unload_model("my-model")
        assert result is True
        mock_proc.terminate.assert_called_once()

    def test_unload_without_process(self):
        backend = LlamaCppBackend()
        backend._server_process = None
        result = backend.unload_model("my-model")
        assert result is True


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    """warmup() sends a short chat and returns bool."""

    def test_warmup_success(self, llamacpp_chat_response, mock_httpx_response):
        backend = LlamaCppBackend()
        mock_resp = mock_httpx_response(json_data=llamacpp_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            assert backend.warmup("my-model", timeout=30.0) is True

    def test_warmup_failure(self):
        backend = LlamaCppBackend()
        with patch.object(
            backend._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            assert backend.warmup("my-model", timeout=30.0) is False


# ---------------------------------------------------------------------------
# detect_context_window
# ---------------------------------------------------------------------------


class TestDetectContextWindow:
    """detect_context_window() reads from /props or defaults to 4096."""

    def test_context_window_from_props(self, mock_httpx_response):
        backend = LlamaCppBackend()
        mock_resp = mock_httpx_response(
            json_data={"default_generation_settings": {"n_ctx": 8192}}
        )
        with patch.object(backend._client, "get", return_value=mock_resp):
            assert backend.detect_context_window("my-model") == 8192

    def test_context_window_default(self):
        backend = LlamaCppBackend()
        with patch.object(
            backend._client, "get", side_effect=httpx.ConnectError("fail")
        ):
            assert backend.detect_context_window("my-model") == 4096


# ---------------------------------------------------------------------------
# get_model_size
# ---------------------------------------------------------------------------


class TestGetModelSize:
    """get_model_size() returns None for llama-server."""

    def test_returns_none(self):
        backend = LlamaCppBackend()
        assert backend.get_model_size("my-model") is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """All httpx errors wrapped in BackendError."""

    def test_connect_error_is_retryable(self):
        backend = LlamaCppBackend()
        with patch.object(
            backend._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(BackendError) as exc_info:
                backend.chat("m", [{"role": "user", "content": "hi"}])
            assert exc_info.value.retryable is True

    def test_timeout_error_is_retryable(self):
        backend = LlamaCppBackend()
        with patch.object(
            backend._client, "post", side_effect=httpx.TimeoutException("timeout")
        ):
            with pytest.raises(BackendError) as exc_info:
                backend.chat("m", [{"role": "user", "content": "hi"}])
            assert exc_info.value.retryable is True

    def test_http_status_error_wrapped(self):
        backend = LlamaCppBackend()
        mock_resp = httpx.Response(status_code=500)
        mock_resp._request = httpx.Request("POST", "http://test")
        error = httpx.HTTPStatusError(
            "server error", request=mock_resp._request, response=mock_resp
        )
        with patch.object(backend._client, "post", side_effect=error), pytest.raises(BackendError):
            backend.chat("m", [{"role": "user", "content": "hi"}])
