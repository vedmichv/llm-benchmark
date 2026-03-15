"""Tests for pre-flight checks (backend installation, connectivity, model availability, RAM warnings)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_backend(**kwargs) -> MagicMock:
    """Create a mock Backend with default behavior."""
    backend = MagicMock()
    backend.name = "ollama"
    backend.version = "0.6.1"
    backend.check_connectivity.return_value = True
    backend.list_models.return_value = []
    for k, v in kwargs.items():
        setattr(backend, k, v)
    return backend


@dataclass
class _FakeBackendStatus:
    """Minimal stand-in for detection.BackendStatus in tests."""
    name: str
    installed: bool
    running: bool
    binary_path: str | None
    port: int


def _make_statuses(
    ollama=(True, True),
    llama_cpp=(False, False),
    lm_studio=(False, False),
) -> list[_FakeBackendStatus]:
    """Build a list of fake BackendStatus objects."""
    return [
        _FakeBackendStatus("ollama", ollama[0], ollama[1], "/usr/bin/ollama" if ollama[0] else None, 11434),
        _FakeBackendStatus("llama-cpp", llama_cpp[0], llama_cpp[1], "/usr/bin/llama-server" if llama_cpp[0] else None, 8080),
        _FakeBackendStatus("lm-studio", lm_studio[0], lm_studio[1], "/usr/bin/lms" if lm_studio[0] else None, 1234),
    ]


class TestBackendInstalled:
    """Tests for check_backend_installed()."""

    def test_installed_and_running_returns_true(self):
        """When backend is installed and running, returns True without prompts."""
        from llm_benchmark.preflight import check_backend_installed

        with patch("llm_benchmark.preflight.detect_backends", return_value=_make_statuses(ollama=(True, True))):
            result = check_backend_installed("ollama")

        assert result is True

    def test_not_installed_returns_false(self, capsys):
        """When backend is not installed, shows install instructions and returns False."""
        from llm_benchmark.preflight import check_backend_installed

        with (
            patch("llm_benchmark.preflight.detect_backends", return_value=_make_statuses(llama_cpp=(False, False))),
            patch("llm_benchmark.preflight.get_install_instructions", return_value="brew install llama.cpp"),
        ):
            result = check_backend_installed("llama-cpp")

        assert result is False
        captured = capsys.readouterr()
        assert "not installed" in captured.out
        assert "brew install llama.cpp" in captured.out

    def test_installed_not_running_user_declines(self, capsys):
        """When installed but not running and user declines start, returns False."""
        from llm_benchmark.preflight import check_backend_installed

        with (
            patch("llm_benchmark.preflight.detect_backends", return_value=_make_statuses(lm_studio=(True, False))),
            patch("builtins.input", return_value="n"),
        ):
            result = check_backend_installed("lm-studio")

        assert result is False
        captured = capsys.readouterr()
        assert "not running" in captured.out

    def test_installed_not_running_user_accepts_success(self, capsys):
        """When installed but not running and user accepts, auto-start succeeds."""
        from llm_benchmark.preflight import check_backend_installed

        with (
            patch("llm_benchmark.preflight.detect_backends", return_value=_make_statuses(ollama=(True, False))),
            patch("builtins.input", return_value="y"),
            patch("llm_benchmark.preflight.auto_start_backend") as mock_start,
        ):
            mock_start.return_value = MagicMock()
            result = check_backend_installed("ollama")

        assert result is True
        mock_start.assert_called_once_with("ollama")

    def test_installed_not_running_auto_start_fails(self, capsys):
        """When auto-start fails, returns False with error message."""
        from llm_benchmark.preflight import check_backend_installed

        with (
            patch("llm_benchmark.preflight.detect_backends", return_value=_make_statuses(ollama=(True, False))),
            patch("builtins.input", return_value="y"),
            patch("llm_benchmark.preflight.auto_start_backend", side_effect=Exception("start failed")),
        ):
            result = check_backend_installed("ollama")

        assert result is False
        captured = capsys.readouterr()
        assert "Failed to start" in captured.out

    def test_unknown_backend_returns_false(self, capsys):
        """When backend_name not in detection results, returns False."""
        from llm_benchmark.preflight import check_backend_installed

        with patch("llm_benchmark.preflight.detect_backends", return_value=_make_statuses()):
            result = check_backend_installed("nonexistent")

        assert result is False

    def test_eof_on_input_declines(self):
        """EOFError during input prompt treated as decline."""
        from llm_benchmark.preflight import check_backend_installed

        with (
            patch("llm_benchmark.preflight.detect_backends", return_value=_make_statuses(ollama=(True, False))),
            patch("builtins.input", side_effect=EOFError),
        ):
            result = check_backend_installed("ollama")

        assert result is False


class TestOllamaInstallationBackwardCompat:
    """Backward compatibility: check_ollama_installed() still works."""

    def test_check_ollama_installed_delegates(self):
        """check_ollama_installed() delegates to check_backend_installed('ollama')."""
        from llm_benchmark.preflight import check_ollama_installed

        with patch("llm_benchmark.preflight.check_backend_installed", return_value=True) as mock_check:
            result = check_ollama_installed()

        assert result is True
        mock_check.assert_called_once_with("ollama")


class TestBackendConnectivity:
    """Tests for check_backend_connectivity()."""

    def test_backend_unreachable_ollama(self, capsys):
        """Ollama-specific message when check_connectivity fails."""
        from llm_benchmark.preflight import check_backend_connectivity

        backend = _make_mock_backend(name="ollama")
        backend.check_connectivity.return_value = False
        result = check_backend_connectivity(backend)

        assert result is False
        captured = capsys.readouterr()
        assert "Cannot connect to ollama" in captured.out
        assert "ollama serve" in captured.out

    def test_backend_unreachable_llama_cpp(self, capsys):
        """llama-cpp specific message when check_connectivity fails."""
        from llm_benchmark.preflight import check_backend_connectivity

        backend = _make_mock_backend(name="llama-cpp")
        backend.check_connectivity.return_value = False
        result = check_backend_connectivity(backend)

        assert result is False
        captured = capsys.readouterr()
        assert "Cannot connect to llama-cpp" in captured.out
        assert "llama-server" in captured.out

    def test_backend_unreachable_lm_studio(self, capsys):
        """lm-studio specific message when check_connectivity fails."""
        from llm_benchmark.preflight import check_backend_connectivity

        backend = _make_mock_backend(name="lm-studio")
        backend.check_connectivity.return_value = False
        result = check_backend_connectivity(backend)

        assert result is False
        captured = capsys.readouterr()
        assert "Cannot connect to lm-studio" in captured.out
        assert "lms server start" in captured.out

    def test_backend_reachable(self):
        """When backend.check_connectivity() returns True, returns True."""
        from llm_benchmark.preflight import check_backend_connectivity

        backend = _make_mock_backend()
        backend.check_connectivity.return_value = True
        result = check_backend_connectivity(backend)

        assert result is True

    def test_backend_connectivity_exception(self, capsys):
        """Exception during check_connectivity treated as unreachable."""
        from llm_benchmark.preflight import check_backend_connectivity

        backend = _make_mock_backend(name="ollama")
        backend.check_connectivity.side_effect = ConnectionError("refused")
        result = check_backend_connectivity(backend)

        assert result is False
        captured = capsys.readouterr()
        assert "Cannot connect to ollama" in captured.out


class TestAvailableModels:
    """Tests for check_available_models()."""

    def test_no_models_ollama_hint(self, capsys):
        """Ollama shows 'ollama pull' hint when no models found."""
        from llm_benchmark.preflight import check_available_models

        backend = _make_mock_backend(name="ollama")
        backend.list_models.return_value = []

        result = check_available_models(backend)

        assert result == []
        captured = capsys.readouterr()
        assert "ollama pull llama3.2:1b" in captured.out

    def test_no_models_llama_cpp_hint(self, capsys):
        """llama-cpp shows GGUF download hint when no models found."""
        from llm_benchmark.preflight import check_available_models

        backend = _make_mock_backend(name="llama-cpp")
        backend.list_models.return_value = []

        result = check_available_models(backend)

        assert result == []
        captured = capsys.readouterr()
        assert "huggingface.co" in captured.out

    def test_no_models_lm_studio_hint(self, capsys):
        """lm-studio shows lms load hint when no models found."""
        from llm_benchmark.preflight import check_available_models

        backend = _make_mock_backend(name="lm-studio")
        backend.list_models.return_value = []

        result = check_available_models(backend)

        assert result == []
        captured = capsys.readouterr()
        assert "lms load" in captured.out

    def test_models_found(self):
        """When models exist, returns the model list."""
        from llm_benchmark.preflight import check_available_models

        backend = _make_mock_backend()
        backend.list_models.return_value = [
            {"model": "llama3.2:1b", "size": 1_000_000_000},
            {"model": "gemma:2b", "size": 2_000_000_000},
        ]

        result = check_available_models(backend)

        assert len(result) == 2

    def test_skip_models_filtered(self):
        """Models in skip list are excluded from results."""
        from llm_benchmark.preflight import check_available_models

        backend = _make_mock_backend()
        backend.list_models.return_value = [
            {"model": "llama3.2:1b", "size": 1_000_000_000},
            {"model": "gemma:2b", "size": 2_000_000_000},
        ]

        result = check_available_models(backend, skip_models=["gemma:2b"])

        assert len(result) == 1
        assert result[0]['model'] == "llama3.2:1b"


class TestRamWarning:
    """Tests for check_ram_for_models()."""

    def test_ram_warning(self, capsys):
        """Model estimated to exceed 80% of RAM triggers warning but does not raise."""
        from llm_benchmark.preflight import check_ram_for_models

        model = {"model": "big-model:70b", "size": 7 * 1024 * 1024 * 1024}  # 7GB on disk

        with patch("llm_benchmark.preflight._get_system_ram_gb", return_value=8.0):
            # Should NOT raise -- just warn
            check_ram_for_models([model])

        captured = capsys.readouterr()
        assert "big-model:70b" in captured.out or "warning" in captured.out.lower() or "RAM" in captured.out

    def test_ram_ok(self, capsys):
        """Model within RAM limits produces no warning."""
        from llm_benchmark.preflight import check_ram_for_models

        model = {"model": "small-model:1b", "size": 4 * 1024 * 1024 * 1024}  # 4GB on disk

        with patch("llm_benchmark.preflight._get_system_ram_gb", return_value=32.0):
            check_ram_for_models([model])

        captured = capsys.readouterr()
        # No warning should appear
        assert "warning" not in captured.out.lower()
        assert "RAM" not in captured.out


class TestRunPreflightChecks:
    """Tests for run_preflight_checks()."""

    def test_preflight_exits_on_install_check_failure(self):
        """Exits with code 1 when backend is not installed."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()

        with (
            patch("llm_benchmark.preflight.check_backend_installed", return_value=False),
            patch("llm_benchmark.preflight.check_backend_connectivity") as mock_conn,
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight_checks(backend=backend)
            assert exc_info.value.code == 1
            mock_conn.assert_not_called()

    def test_preflight_exits_on_no_connectivity(self):
        """Exits with code 1 when backend is unreachable."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()

        with (
            patch("llm_benchmark.preflight.check_backend_installed", return_value=True),
            patch("llm_benchmark.preflight.check_backend_connectivity", return_value=False),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight_checks(backend=backend)
            assert exc_info.value.code == 1

    def test_preflight_exits_on_no_models(self):
        """Exits with code 1 when no models are available."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()

        with (
            patch("llm_benchmark.preflight.check_backend_installed", return_value=True),
            patch("llm_benchmark.preflight.check_backend_connectivity", return_value=True),
            patch("llm_benchmark.preflight.check_available_models", return_value=[]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight_checks(backend=backend)
            assert exc_info.value.code == 1

    def test_preflight_skips_ram_check_when_skip_checks(self):
        """With skip_checks=True, RAM check is skipped."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()
        model = {"model": "test:1b", "size": 1_000_000_000}

        with (
            patch("llm_benchmark.preflight.check_backend_installed", return_value=True),
            patch("llm_benchmark.preflight.check_backend_connectivity", return_value=True),
            patch("llm_benchmark.preflight.check_available_models", return_value=[model]),
            patch("llm_benchmark.preflight.check_ram_for_models") as mock_ram,
        ):
            result = run_preflight_checks(backend=backend, skip_checks=True)
            mock_ram.assert_not_called()
            assert len(result) == 1

    def test_preflight_uses_backend_name_for_install_check(self):
        """run_preflight_checks passes backend.name to check_backend_installed."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend(name="llama-cpp")

        with (
            patch("llm_benchmark.preflight.check_backend_installed", return_value=False) as mock_install,
        ):
            with pytest.raises(SystemExit):
                run_preflight_checks(backend=backend)
            mock_install.assert_called_once_with("llama-cpp")

    def test_preflight_full_chain_success(self):
        """Full preflight chain succeeds and returns models."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()
        model = {"model": "test:1b", "size": 1_000_000_000}

        with (
            patch("llm_benchmark.preflight.check_backend_installed", return_value=True),
            patch("llm_benchmark.preflight.check_backend_connectivity", return_value=True),
            patch("llm_benchmark.preflight.check_available_models", return_value=[model]),
            patch("llm_benchmark.preflight.check_ram_for_models"),
        ):
            result = run_preflight_checks(backend=backend)
            assert result == [model]
