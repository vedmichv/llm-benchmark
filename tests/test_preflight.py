"""Tests for pre-flight checks (backend connectivity, model availability, RAM warnings)."""

from __future__ import annotations

import platform
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


class TestOllamaInstallation:
    """Tests for check_ollama_installed()."""

    def test_binary_found_returns_true(self, capsys):
        """When shutil.which('ollama') returns a path, returns True without output."""
        from llm_benchmark.preflight import check_ollama_installed

        with patch("llm_benchmark.preflight.shutil.which", return_value="/usr/local/bin/ollama"):
            result = check_ollama_installed()

        assert result is True
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_binary_not_found_user_declines(self, capsys):
        """When binary missing and user inputs 'n', returns False with platform-specific command."""
        from llm_benchmark.preflight import check_ollama_installed

        with (
            patch("llm_benchmark.preflight.shutil.which", return_value=None),
            patch("llm_benchmark.preflight.platform.system", return_value="Darwin"),
            patch("builtins.input", return_value="n"),
        ):
            result = check_ollama_installed()

        assert result is False
        captured = capsys.readouterr()
        assert "not installed" in captured.out.lower() or "Ollama is not installed" in captured.out
        assert "curl -fsSL https://ollama.com/install.sh | sh" in captured.out

    def test_binary_not_found_user_declines_windows(self, capsys):
        """Windows shows PowerShell install command."""
        from llm_benchmark.preflight import check_ollama_installed

        with (
            patch("llm_benchmark.preflight.shutil.which", return_value=None),
            patch("llm_benchmark.preflight.platform.system", return_value="Windows"),
            patch("builtins.input", return_value="n"),
        ):
            result = check_ollama_installed()

        assert result is False
        captured = capsys.readouterr()
        assert "irm https://ollama.com/install.ps1 | iex" in captured.out

    def test_binary_not_found_user_declines_linux(self, capsys):
        """Linux shows curl install command."""
        from llm_benchmark.preflight import check_ollama_installed

        with (
            patch("llm_benchmark.preflight.shutil.which", return_value=None),
            patch("llm_benchmark.preflight.platform.system", return_value="Linux"),
            patch("builtins.input", return_value="n"),
        ):
            result = check_ollama_installed()

        assert result is False
        captured = capsys.readouterr()
        assert "curl -fsSL https://ollama.com/install.sh | sh" in captured.out

    def test_binary_not_found_user_accepts_success(self, capsys):
        """When user accepts and install succeeds, returns True."""
        from llm_benchmark.preflight import check_ollama_installed

        with (
            patch("llm_benchmark.preflight.shutil.which", side_effect=[None, "/usr/local/bin/ollama"]),
            patch("llm_benchmark.preflight.platform.system", return_value="Darwin"),
            patch("builtins.input", return_value="y"),
            patch("llm_benchmark.preflight.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = check_ollama_installed()

        assert result is True
        captured = capsys.readouterr()
        assert "installed successfully" in captured.out.lower()

    def test_binary_not_found_user_accepts_failure(self, capsys):
        """When user accepts but install fails, returns False."""
        from llm_benchmark.preflight import check_ollama_installed

        with (
            patch("llm_benchmark.preflight.shutil.which", side_effect=[None, None]),
            patch("llm_benchmark.preflight.platform.system", return_value="Darwin"),
            patch("builtins.input", return_value="y"),
            patch("llm_benchmark.preflight.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1)
            result = check_ollama_installed()

        assert result is False
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower()

    def test_preflight_calls_install_check_first(self):
        """run_preflight_checks calls check_ollama_installed before connectivity check."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()

        with (
            patch("llm_benchmark.preflight.check_ollama_installed", return_value=False) as mock_install,
            patch("llm_benchmark.preflight.check_backend_connectivity") as mock_conn,
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight_checks(backend=backend)
            assert exc_info.value.code == 1
            mock_install.assert_called_once()
            mock_conn.assert_not_called()


class TestBackendConnectivity:
    """Tests for check_backend_connectivity()."""

    def test_backend_unreachable(self, capsys):
        """When backend.check_connectivity() returns False, returns False with guidance."""
        from llm_benchmark.preflight import check_backend_connectivity

        backend = _make_mock_backend()
        backend.check_connectivity.return_value = False
        result = check_backend_connectivity(backend)

        assert result is False
        captured = capsys.readouterr()
        # Should contain platform-specific instruction
        os_name = platform.system()
        if os_name == "Darwin":
            assert "ollama serve" in captured.out
        elif os_name == "Windows":
            assert "Start the Ollama application" in captured.out
        else:
            assert "ollama serve" in captured.out
        # Should contain download link
        assert "ollama.com" in captured.out

    def test_backend_reachable(self):
        """When backend.check_connectivity() returns True, returns True."""
        from llm_benchmark.preflight import check_backend_connectivity

        backend = _make_mock_backend()
        backend.check_connectivity.return_value = True
        result = check_backend_connectivity(backend)

        assert result is True


class TestAvailableModels:
    """Tests for check_available_models()."""

    def test_no_models_found(self, capsys):
        """When no models exist, returns empty list and suggests pulling a model."""
        from llm_benchmark.preflight import check_available_models

        backend = _make_mock_backend()
        backend.list_models.return_value = []

        result = check_available_models(backend)

        assert result == []
        captured = capsys.readouterr()
        assert "ollama pull llama3.2:1b" in captured.out

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

    def test_preflight_exits_on_no_connectivity(self):
        """Exits with code 1 when backend is unreachable."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()

        with patch("llm_benchmark.preflight.check_ollama_installed", return_value=True), \
             patch("llm_benchmark.preflight.check_backend_connectivity", return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight_checks(backend=backend)
            assert exc_info.value.code == 1

    def test_preflight_exits_on_no_models(self):
        """Exits with code 1 when no models are available."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()

        with patch("llm_benchmark.preflight.check_ollama_installed", return_value=True), \
             patch("llm_benchmark.preflight.check_backend_connectivity", return_value=True), \
             patch("llm_benchmark.preflight.check_available_models", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight_checks(backend=backend)
            assert exc_info.value.code == 1

    def test_preflight_skips_ram_check_when_skip_checks(self):
        """With skip_checks=True, RAM check is skipped."""
        from llm_benchmark.preflight import run_preflight_checks

        backend = _make_mock_backend()
        model = {"model": "test:1b", "size": 1_000_000_000}

        with patch("llm_benchmark.preflight.check_ollama_installed", return_value=True), \
             patch("llm_benchmark.preflight.check_backend_connectivity", return_value=True), \
             patch("llm_benchmark.preflight.check_available_models", return_value=[model]), \
             patch("llm_benchmark.preflight.check_ram_for_models") as mock_ram:
            result = run_preflight_checks(backend=backend, skip_checks=True)
            mock_ram.assert_not_called()
            assert len(result) == 1
