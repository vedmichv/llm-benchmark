"""Tests for backend abstraction layer: Protocol, models, factory, OllamaBackend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_benchmark.backends import (
    Backend,
    BackendError,
    BackendResponse,
    StreamResult,
    create_backend,
)
from llm_benchmark.backends.ollama import OllamaBackend

# ---------------------------------------------------------------------------
# BackendResponse
# ---------------------------------------------------------------------------


class TestBackendResponse:
    def test_creation_with_seconds(self):
        resp = BackendResponse(
            model="test",
            content="hello",
            done=True,
            eval_duration=4.0,
            eval_count=100,
            prompt_eval_duration=1.0,
            prompt_eval_count=20,
            total_duration=6.0,
            load_duration=1.0,
        )
        assert resp.eval_duration == 4.0
        assert resp.eval_count == 100
        assert resp.model == "test"
        assert resp.content == "hello"
        assert resp.done is True
        assert resp.prompt_cached is False

    def test_prompt_cached_flag(self):
        resp = BackendResponse(
            model="test",
            content="hi",
            done=True,
            eval_duration=1.0,
            eval_count=10,
            prompt_eval_duration=0.0,
            prompt_eval_count=0,
            total_duration=2.0,
            load_duration=0.0,
            prompt_cached=True,
        )
        assert resp.prompt_cached is True
        assert resp.prompt_eval_count == 0


# ---------------------------------------------------------------------------
# BackendError
# ---------------------------------------------------------------------------


class TestBackendError:
    def test_retryable_true(self):
        err = BackendError("fail", retryable=True)
        assert err.retryable is True
        assert str(err) == "fail"

    def test_retryable_false(self):
        err = BackendError("fail", retryable=False)
        assert err.retryable is False

    def test_original_exception(self):
        orig = ValueError("x")
        err = BackendError("fail", retryable=False, original=orig)
        assert err.original is orig

    def test_original_default_none(self):
        err = BackendError("fail", retryable=True)
        assert err.original is None


# ---------------------------------------------------------------------------
# StreamResult
# ---------------------------------------------------------------------------


class TestStreamResult:
    def test_chunks_iteration(self):
        chunks_data = ["Hello", " ", "world"]

        def make_chunks():
            yield from chunks_data

        final_resp = BackendResponse(
            model="test",
            content="Hello world",
            done=True,
            eval_duration=1.0,
            eval_count=3,
            prompt_eval_duration=0.5,
            prompt_eval_count=5,
            total_duration=2.0,
            load_duration=0.5,
        )

        result = StreamResult(chunks=make_chunks(), finalize=lambda: final_resp)
        collected = list(result.chunks)
        assert collected == ["Hello", " ", "world"]

    def test_response_after_iteration(self):
        def make_chunks():
            yield "a"
            yield "b"

        final_resp = BackendResponse(
            model="test",
            content="ab",
            done=True,
            eval_duration=1.0,
            eval_count=2,
            prompt_eval_duration=0.5,
            prompt_eval_count=5,
            total_duration=2.0,
            load_duration=0.5,
        )

        result = StreamResult(chunks=make_chunks(), finalize=lambda: final_resp)
        # Consume chunks first
        list(result.chunks)
        resp = result.response
        assert isinstance(resp, BackendResponse)
        assert resp.content == "ab"
        assert resp.eval_count == 2


# ---------------------------------------------------------------------------
# create_backend factory
# ---------------------------------------------------------------------------


class TestCreateBackend:
    def test_create_ollama(self):
        backend = create_backend("ollama")
        assert isinstance(backend, OllamaBackend)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            create_backend("unknown")


# ---------------------------------------------------------------------------
# OllamaBackend Protocol compliance
# ---------------------------------------------------------------------------


class TestOllamaBackendProtocol:
    def test_isinstance_check(self):
        backend = OllamaBackend()
        assert isinstance(backend, Backend)

    def test_name_property(self):
        backend = OllamaBackend()
        assert backend.name == "ollama"


# ---------------------------------------------------------------------------
# OllamaBackend._to_response()
# ---------------------------------------------------------------------------


class TestOllamaBackendToResponse:
    def test_ns_to_seconds_conversion(self):
        raw = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "total_duration": 5_000_000_000,
            "load_duration": 1_000_000_000,
            "prompt_eval_duration": 500_000_000,
            "prompt_eval_count": 10,
            "eval_duration": 3_000_000_000,
            "eval_count": 50,
        }
        backend = OllamaBackend()
        resp = backend._to_response(raw)
        assert isinstance(resp, BackendResponse)
        assert resp.total_duration == 5.0
        assert resp.load_duration == 1.0
        assert resp.prompt_eval_duration == 0.5
        assert resp.eval_duration == 3.0
        assert resp.eval_count == 50
        assert resp.prompt_eval_count == 10
        assert resp.content == "hi"
        assert resp.prompt_cached is False

    def test_prompt_caching_detection(self):
        raw = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "cached"},
            "done": True,
            "total_duration": 2_000_000_000,
            "load_duration": 0,
            "prompt_eval_duration": 0,
            "prompt_eval_count": -1,
            "eval_duration": 1_000_000_000,
            "eval_count": 20,
        }
        backend = OllamaBackend()
        resp = backend._to_response(raw)
        assert resp.prompt_cached is True
        assert resp.prompt_eval_count == 0


# ---------------------------------------------------------------------------
# OllamaBackend.chat() error wrapping
# ---------------------------------------------------------------------------


class TestOllamaBackendChatErrors:
    @patch("llm_benchmark.backends.ollama.ollama")
    def test_request_error_retryable(self, mock_ollama):
        import ollama as _ollama

        mock_ollama.chat.side_effect = _ollama.RequestError("bad request")
        backend = OllamaBackend()
        with pytest.raises(BackendError) as exc_info:
            backend.chat("model", [{"role": "user", "content": "hi"}])
        assert exc_info.value.retryable is True

    @patch("llm_benchmark.backends.ollama.ollama")
    def test_response_error_500_retryable(self, mock_ollama):
        import ollama as _ollama

        error = _ollama.ResponseError("server error")
        error.status_code = 500
        mock_ollama.chat.side_effect = error
        backend = OllamaBackend()
        with pytest.raises(BackendError) as exc_info:
            backend.chat("model", [{"role": "user", "content": "hi"}])
        assert exc_info.value.retryable is True

    @patch("llm_benchmark.backends.ollama.ollama")
    def test_response_error_404_not_retryable(self, mock_ollama):
        import ollama as _ollama

        error = _ollama.ResponseError("not found")
        error.status_code = 404
        mock_ollama.chat.side_effect = error
        backend = OllamaBackend()
        with pytest.raises(BackendError) as exc_info:
            backend.chat("model", [{"role": "user", "content": "hi"}])
        assert exc_info.value.retryable is False

    @patch("llm_benchmark.backends.ollama.ollama")
    def test_connection_error_retryable(self, mock_ollama):
        mock_ollama.chat.side_effect = ConnectionError("refused")
        backend = OllamaBackend()
        with pytest.raises(BackendError) as exc_info:
            backend.chat("model", [{"role": "user", "content": "hi"}])
        assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# OllamaBackend.chat() non-streaming
# ---------------------------------------------------------------------------


class TestOllamaBackendChat:
    @patch("llm_benchmark.backends.ollama.ollama")
    def test_chat_returns_backend_response(self, mock_ollama):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "answer"},
            "done": True,
            "total_duration": 4_000_000_000,
            "load_duration": 500_000_000,
            "prompt_eval_duration": 200_000_000,
            "prompt_eval_count": 5,
            "eval_duration": 3_000_000_000,
            "eval_count": 40,
        }
        mock_ollama.chat.return_value = mock_response
        backend = OllamaBackend()
        resp = backend.chat("llama3", [{"role": "user", "content": "hi"}])
        assert isinstance(resp, BackendResponse)
        assert resp.model == "llama3"
        assert resp.content == "answer"
        assert resp.eval_duration == 3.0


# ---------------------------------------------------------------------------
# OllamaBackend.chat(stream=True)
# ---------------------------------------------------------------------------


class TestOllamaBackendStream:
    @patch("llm_benchmark.backends.ollama.ollama")
    def test_stream_returns_stream_result(self, mock_ollama):
        # Simulate streaming: intermediate chunks have content, final has timing
        chunk1 = {"message": {"content": "He"}, "done": False}
        chunk2 = {"message": {"content": "llo"}, "done": False}
        final_chunk = {
            "model": "llama3",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "total_duration": 2_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_duration": 300_000_000,
            "prompt_eval_count": 3,
            "eval_duration": 1_500_000_000,
            "eval_count": 2,
        }

        mock_ollama.chat.return_value = iter([
            MagicMock(**{"get": lambda k, d=None, _c=chunk1: _c.get(k, d)}),
            MagicMock(**{"get": lambda k, d=None, _c=chunk2: _c.get(k, d)}),
            MagicMock(**{"get": lambda k, d=None, _c=final_chunk: _c.get(k, d)}),
        ])

        # We need dict-like access, let's use simple dicts
        mock_ollama.chat.return_value = iter([chunk1, chunk2, final_chunk])

        backend = OllamaBackend()
        result = backend.chat(
            "llama3", [{"role": "user", "content": "hi"}], stream=True
        )
        assert isinstance(result, StreamResult)

        chunks = list(result.chunks)
        assert "He" in chunks
        assert "llo" in chunks

        resp = result.response
        assert isinstance(resp, BackendResponse)
        assert resp.eval_count == 2
