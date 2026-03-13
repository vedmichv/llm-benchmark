"""Tests for the parameter sweep module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_benchmark.models import SweepConfigResult, SweepModelResult
from llm_benchmark.sweep import (
    build_sweep_configs,
    get_model_layers,
    run_sweep_for_model,
)


class TestGetModelLayers:
    """Tests for get_model_layers()."""

    @patch("llm_benchmark.sweep.ollama")
    def test_returns_block_count(self, mock_ollama: MagicMock) -> None:
        """Should extract block_count from modelinfo keys."""
        mock_response = MagicMock()
        mock_response.modelinfo = {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.context_length": 4096,
        }
        mock_ollama.show.return_value = mock_response
        assert get_model_layers("llama3:8b") == 32

    @patch("llm_benchmark.sweep.ollama")
    def test_returns_none_when_missing(self, mock_ollama: MagicMock) -> None:
        """Should return None when no block_count key exists."""
        mock_response = MagicMock()
        mock_response.modelinfo = {
            "general.architecture": "llama",
        }
        mock_ollama.show.return_value = mock_response
        assert get_model_layers("some-model") is None

    @patch("llm_benchmark.sweep.ollama")
    def test_returns_none_on_exception(self, mock_ollama: MagicMock) -> None:
        """Should return None if ollama.show() raises."""
        mock_ollama.show.side_effect = Exception("Connection refused")
        assert get_model_layers("bad-model") is None


class TestBuildSweepConfigs:
    """Tests for build_sweep_configs()."""

    def test_with_gpu_and_block_count(self) -> None:
        """With GPU + block_count=32, should produce 4 num_ctx * 3 num_gpu = 12."""
        configs = build_sweep_configs(block_count=32, has_gpu=True)
        assert len(configs) == 12
        # All 4 num_ctx values present
        num_ctx_values = sorted(set(c[0] for c in configs))
        assert num_ctx_values == [512, 1024, 2048, 4096]
        # 3 num_gpu values: 0, 16, 32
        num_gpu_values = sorted(set(c[1] for c in configs))
        assert num_gpu_values == [0, 16, 32]

    def test_no_gpu(self) -> None:
        """Without GPU, should produce 4 configs with num_gpu=0."""
        configs = build_sweep_configs(block_count=None, has_gpu=False)
        assert len(configs) == 4
        assert all(c[1] == 0 for c in configs)

    def test_no_block_count_with_gpu(self) -> None:
        """GPU detected but no block_count => only num_ctx variations."""
        configs = build_sweep_configs(block_count=None, has_gpu=True)
        assert len(configs) == 4
        assert all(c[1] == 0 for c in configs)

    def test_block_count_no_gpu(self) -> None:
        """Block count known but no GPU => only num_ctx variations."""
        configs = build_sweep_configs(block_count=32, has_gpu=False)
        assert len(configs) == 4
        assert all(c[1] == 0 for c in configs)


class TestBestConfigSelection:
    """Tests for best config selection logic."""

    def test_best_config_highest_response_ts(self) -> None:
        """best_config should be the config with highest response_ts."""
        configs = [
            SweepConfigResult(
                model="test",
                num_ctx=512,
                num_gpu=0,
                response_ts=10.0,
                total_ts=8.0,
                eval_count=50,
                total_duration_s=5.0,
                success=True,
            ),
            SweepConfigResult(
                model="test",
                num_ctx=2048,
                num_gpu=0,
                response_ts=25.0,
                total_ts=20.0,
                eval_count=50,
                total_duration_s=2.0,
                success=True,
            ),
            SweepConfigResult(
                model="test",
                num_ctx=4096,
                num_gpu=0,
                response_ts=15.0,
                total_ts=12.0,
                eval_count=50,
                total_duration_s=3.0,
                success=True,
            ),
        ]
        result = SweepModelResult(
            model="test",
            configs=configs,
            best_config=max(
                (c for c in configs if c.success), key=lambda c: c.response_ts
            ),
        )
        assert result.best_config is not None
        assert result.best_config.num_ctx == 2048
        assert result.best_config.response_ts == 25.0


class TestFailedConfigContinues:
    """Tests that failed configs don't stop the sweep."""

    @patch("llm_benchmark.sweep._get_gpu_info")
    @patch("llm_benchmark.sweep.get_model_layers")
    @patch("llm_benchmark.sweep.unload_model")
    @patch("llm_benchmark.sweep.warmup_model")
    @patch("llm_benchmark.sweep.ollama")
    def test_failed_config_recorded(
        self,
        mock_ollama: MagicMock,
        mock_warmup: MagicMock,
        mock_unload: MagicMock,
        mock_get_layers: MagicMock,
        mock_gpu_info: MagicMock,
    ) -> None:
        """A config that raises should be recorded with success=False."""
        mock_gpu_info.return_value = ("No dedicated GPU", None)
        mock_get_layers.return_value = None  # No GPU layers
        mock_warmup.return_value = True
        mock_unload.return_value = True

        call_count = 0

        def chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("CUDA out of memory")
            # Return a mock response for successful calls
            resp = MagicMock()
            resp.eval_count = 50
            resp.eval_duration = 1_000_000_000  # 1 second
            resp.total_duration = 2_000_000_000
            resp.prompt_eval_count = 10
            resp.prompt_eval_duration = 500_000_000
            resp.load_duration = 100_000_000
            resp.model = "test"
            resp.created_at = "2024-01-01T00:00:00Z"
            resp.done = True
            resp.message = MagicMock()
            resp.message.role = "assistant"
            resp.message.content = "Test response"
            return resp

        mock_ollama.chat.side_effect = chat_side_effect
        mock_ollama.Options = MagicMock(side_effect=lambda **kw: kw)

        result = run_sweep_for_model("test-model", timeout=30, skip_warmup=True)

        # Should have 4 configs (no GPU = num_ctx only)
        assert len(result.configs) == 4
        # One failed
        failed = [c for c in result.configs if not c.success]
        assert len(failed) == 1
        assert "CUDA out of memory" in failed[0].error
        # Three succeeded
        succeeded = [c for c in result.configs if c.success]
        assert len(succeeded) == 3
        # Best config exists (from successful ones)
        assert result.best_config is not None
        assert result.best_config.success is True
