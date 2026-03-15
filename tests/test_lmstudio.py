"""Tests for the LMStudioBackend implementation."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_benchmark.backends import Backend, BackendError, BackendResponse, StreamResult
from llm_benchmark.backends.lmstudio import LMStudioBackend


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """LMStudioBackend satisfies the Backend Protocol."""

    def test_isinstance_backend(self):
        backend = LMStudioBackend()
        assert isinstance(backend, Backend)

    def test_name_property(self):
        backend = LMStudioBackend()
        assert backend.name == "lm-studio"

    def test_version_property_returns_string(self):
        backend = LMStudioBackend()
        assert isinstance(backend.version, str)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Constructor accepts host and port with defaults."""

    def test_default_host_port(self):
        backend = LMStudioBackend()
        assert "127.0.0.1" in backend._base_url
        assert "1234" in backend._base_url

    def test_custom_host_port(self):
        backend = LMStudioBackend(host="10.0.0.1", port=5555)
        assert "10.0.0.1" in backend._base_url
        assert "5555" in backend._base_url


# ---------------------------------------------------------------------------
# chat() non-streaming
# ---------------------------------------------------------------------------


class TestChatNonStreaming:
    """chat() sends POST and returns BackendResponse with timing derived from tps."""

    def test_chat_returns_backend_response(self, lmstudio_chat_response, mock_httpx_response):
        backend = LMStudioBackend()
        mock_resp = mock_httpx_response(json_data=lmstudio_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("my-model", [{"role": "user", "content": "Hi"}])
        assert isinstance(result, BackendResponse)

    def test_chat_timing_from_tokens_per_second(self, lmstudio_chat_response, mock_httpx_response):
        """eval_duration derived from eval_count / tokens_per_second."""
        backend = LMStudioBackend()
        mock_resp = mock_httpx_response(json_data=lmstudio_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("my-model", [{"role": "user", "content": "Hi"}])
        assert result.eval_count == 35
        assert result.prompt_eval_count == 10
        # eval_duration = 35 / 45.2
        assert result.eval_duration == pytest.approx(35 / 45.2)

    def test_chat_content_extracted(self, lmstudio_chat_response, mock_httpx_response):
        backend = LMStudioBackend()
        mock_resp = mock_httpx_response(json_data=lmstudio_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("my-model", [{"role": "user", "content": "Hi"}])
        assert "scattering" in result.content

    def test_chat_model_field(self, lmstudio_chat_response, mock_httpx_response):
        backend = LMStudioBackend()
        mock_resp = mock_httpx_response(json_data=lmstudio_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("my-model", [{"role": "user", "content": "Hi"}])
        assert result.model == "my-lmstudio-model"

    def test_chat_zero_tps_uses_elapsed_time(self, mock_httpx_response):
        """When tokens_per_second is 0, eval_duration falls back to elapsed time."""
        backend = LMStudioBackend()
        data = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "model": "m",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "stats": {"tokens_per_second": 0},
        }
        mock_resp = mock_httpx_response(json_data=data)
        with patch.object(backend._client, "post", return_value=mock_resp):
            result = backend.chat("m", [{"role": "user", "content": "hi"}])
        assert result.eval_duration > 0.0
        assert result.eval_count == 10


# ---------------------------------------------------------------------------
# chat() streaming
# ---------------------------------------------------------------------------


class TestChatStreaming:
    """chat(stream=True) returns StreamResult with accumulated chunks."""

    def _make_sse_lines(self):
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}],"model":"m"}',
            'data: {"choices":[{"delta":{"content":" there"}}],"model":"m"}',
            'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"model":"m","usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7},"stats":{"tokens_per_second":50.0}}',
            "data: [DONE]",
        ]
        return chunks

    def test_stream_returns_stream_result(self):
        backend = LMStudioBackend()
        lines = self._make_sse_lines()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch.object(backend._client, "stream") as mock_stream:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_resp)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_cm

            result = backend.chat("m", [{"role": "user", "content": "Hi"}], stream=True)
        assert isinstance(result, StreamResult)

    def test_stream_chunks_yield_content(self):
        backend = LMStudioBackend()
        lines = self._make_sse_lines()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch.object(backend._client, "stream") as mock_stream:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_resp)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_cm

            result = backend.chat("m", [{"role": "user", "content": "Hi"}], stream=True)
            parts = list(result.chunks)
        assert "Hello" in parts
        assert " there" in parts

    def test_stream_response_has_timings(self):
        backend = LMStudioBackend()
        lines = self._make_sse_lines()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch.object(backend._client, "stream") as mock_stream:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_resp)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_cm

            result = backend.chat("m", [{"role": "user", "content": "Hi"}], stream=True)
            list(result.chunks)
            resp = result.response
        assert resp.eval_count == 2
        assert resp.eval_duration == pytest.approx(2 / 50.0)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    """list_models() queries /api/v1/models for full catalog."""

    def test_list_models_returns_list(self, mock_httpx_response):
        backend = LMStudioBackend()
        resp_data = {
            "data": [
                {"id": "model-a", "object": "model"},
                {"id": "model-b", "object": "model"},
            ]
        }
        mock_resp = mock_httpx_response(json_data=resp_data)
        with patch.object(backend._client, "get", return_value=mock_resp):
            models = backend.list_models()
        assert len(models) == 2
        assert models[0]["model"] == "model-a"
        assert models[1]["model"] == "model-b"


# ---------------------------------------------------------------------------
# unload_model
# ---------------------------------------------------------------------------


class TestUnloadModel:
    """unload_model() sends POST to /api/v1/models/unload."""

    def test_unload_success(self, mock_httpx_response):
        backend = LMStudioBackend()
        mock_resp = mock_httpx_response(json_data={"success": True})
        with patch.object(backend._client, "post", return_value=mock_resp):
            assert backend.unload_model("model-a") is True

    def test_unload_failure(self):
        backend = LMStudioBackend()
        with patch.object(
            backend._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            assert backend.unload_model("model-a") is False


# ---------------------------------------------------------------------------
# check_connectivity
# ---------------------------------------------------------------------------


class TestCheckConnectivity:
    """check_connectivity() checks /v1/models endpoint."""

    def test_connectivity_true_on_200(self, mock_httpx_response):
        backend = LMStudioBackend()
        mock_resp = mock_httpx_response(json_data={"data": []})
        with patch.object(backend._client, "get", return_value=mock_resp):
            assert backend.check_connectivity() is True

    def test_connectivity_false_on_error(self):
        backend = LMStudioBackend()
        with patch.object(
            backend._client, "get", side_effect=httpx.ConnectError("refused")
        ):
            assert backend.check_connectivity() is False


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    """warmup() sends a short chat and returns bool."""

    def test_warmup_success(self, lmstudio_chat_response, mock_httpx_response):
        backend = LMStudioBackend()
        mock_resp = mock_httpx_response(json_data=lmstudio_chat_response)
        with patch.object(backend._client, "post", return_value=mock_resp):
            assert backend.warmup("my-model", timeout=30.0) is True

    def test_warmup_failure(self):
        backend = LMStudioBackend()
        with patch.object(
            backend._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            assert backend.warmup("my-model", timeout=30.0) is False


# ---------------------------------------------------------------------------
# detect_context_window
# ---------------------------------------------------------------------------


class TestDetectContextWindow:
    """detect_context_window() defaults to 4096."""

    def test_returns_default(self):
        backend = LMStudioBackend()
        assert backend.detect_context_window("my-model") == 4096


# ---------------------------------------------------------------------------
# get_model_size
# ---------------------------------------------------------------------------


class TestGetModelSize:
    """get_model_size() returns None for LM Studio."""

    def test_returns_none(self):
        backend = LMStudioBackend()
        assert backend.get_model_size("my-model") is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """All httpx errors wrapped in BackendError."""

    def test_connect_error_is_retryable(self):
        backend = LMStudioBackend()
        with patch.object(
            backend._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(BackendError) as exc_info:
                backend.chat("m", [{"role": "user", "content": "hi"}])
            assert exc_info.value.retryable is True

    def test_timeout_error_is_retryable(self):
        backend = LMStudioBackend()
        with patch.object(
            backend._client, "post", side_effect=httpx.TimeoutException("timeout")
        ):
            with pytest.raises(BackendError) as exc_info:
                backend.chat("m", [{"role": "user", "content": "hi"}])
            assert exc_info.value.retryable is True

    def test_http_status_error_wrapped(self):
        backend = LMStudioBackend()
        mock_resp = httpx.Response(status_code=500)
        mock_resp._request = httpx.Request("POST", "http://test")
        error = httpx.HTTPStatusError(
            "server error", request=mock_resp._request, response=mock_resp
        )
        with patch.object(backend._client, "post", side_effect=error):
            with pytest.raises(BackendError):
                backend.chat("m", [{"role": "user", "content": "hi"}])
