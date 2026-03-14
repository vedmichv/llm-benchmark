"""Tests for backend detection, auto-start, and install instructions."""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest


class TestBackendStatus:
    """Tests for BackendStatus dataclass."""

    def test_backend_status_fields(self):
        from llm_benchmark.backends.detection import BackendStatus

        status = BackendStatus(
            name="ollama",
            installed=True,
            running=False,
            binary_path="/usr/local/bin/ollama",
            port=11434,
        )
        assert status.name == "ollama"
        assert status.installed is True
        assert status.running is False
        assert status.binary_path == "/usr/local/bin/ollama"
        assert status.port == 11434

    def test_backend_status_none_binary_path(self):
        from llm_benchmark.backends.detection import BackendStatus

        status = BackendStatus(
            name="llama-cpp",
            installed=False,
            running=False,
            binary_path=None,
            port=8080,
        )
        assert status.binary_path is None
        assert status.installed is False


class TestPortIsOpen:
    """Tests for _port_is_open helper."""

    @patch("llm_benchmark.backends.detection.socket.socket")
    def test_port_open(self, mock_socket_cls):
        from llm_benchmark.backends.detection import _port_is_open

        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect_ex.return_value = 0

        assert _port_is_open("127.0.0.1", 11434) is True
        mock_sock.settimeout.assert_called_once_with(1.0)
        mock_sock.connect_ex.assert_called_once_with(("127.0.0.1", 11434))
        mock_sock.close.assert_called_once()

    @patch("llm_benchmark.backends.detection.socket.socket")
    def test_port_closed(self, mock_socket_cls):
        from llm_benchmark.backends.detection import _port_is_open

        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect_ex.return_value = 111  # Connection refused

        assert _port_is_open("127.0.0.1", 8080) is False
        mock_sock.close.assert_called_once()

    @patch("llm_benchmark.backends.detection.socket.socket")
    def test_port_custom_timeout(self, mock_socket_cls):
        from llm_benchmark.backends.detection import _port_is_open

        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect_ex.return_value = 0

        _port_is_open("127.0.0.1", 1234, timeout=5.0)
        mock_sock.settimeout.assert_called_once_with(5.0)


class TestDetectBackends:
    """Tests for detect_backends()."""

    @patch("llm_benchmark.backends.detection._port_is_open")
    @patch("llm_benchmark.backends.detection.shutil.which")
    def test_all_installed_all_running(self, mock_which, mock_port):
        from llm_benchmark.backends.detection import detect_backends

        mock_which.side_effect = lambda b: {
            "ollama": "/usr/local/bin/ollama",
            "llama-server": "/usr/local/bin/llama-server",
            "lms": "/usr/local/bin/lms",
        }.get(b)
        mock_port.return_value = True

        statuses = detect_backends()
        assert len(statuses) == 3

        ollama = statuses[0]
        assert ollama.name == "ollama"
        assert ollama.installed is True
        assert ollama.running is True
        assert ollama.binary_path == "/usr/local/bin/ollama"
        assert ollama.port == 11434

        llama = statuses[1]
        assert llama.name == "llama-cpp"
        assert llama.installed is True
        assert llama.running is True
        assert llama.binary_path == "/usr/local/bin/llama-server"
        assert llama.port == 8080

        lms = statuses[2]
        assert lms.name == "lm-studio"
        assert lms.installed is True
        assert lms.running is True
        assert lms.binary_path == "/usr/local/bin/lms"
        assert lms.port == 1234

    @patch("llm_benchmark.backends.detection._port_is_open")
    @patch("llm_benchmark.backends.detection.shutil.which")
    def test_none_installed(self, mock_which, mock_port):
        from llm_benchmark.backends.detection import detect_backends

        mock_which.return_value = None
        mock_port.return_value = False

        statuses = detect_backends()
        assert len(statuses) == 3

        for s in statuses:
            assert s.installed is False
            assert s.running is False
            assert s.binary_path is None

    @patch("llm_benchmark.backends.detection._port_is_open")
    @patch("llm_benchmark.backends.detection.shutil.which")
    def test_mixed_installed_running(self, mock_which, mock_port):
        from llm_benchmark.backends.detection import detect_backends

        mock_which.side_effect = lambda b: {
            "ollama": "/usr/local/bin/ollama",
            "llama-server": None,
            "lms": "/usr/local/bin/lms",
        }.get(b)
        mock_port.side_effect = lambda host, port: port == 11434

        statuses = detect_backends()

        ollama = statuses[0]
        assert ollama.installed is True
        assert ollama.running is True

        llama = statuses[1]
        assert llama.installed is False
        assert llama.running is False

        lms = statuses[2]
        assert lms.installed is True
        assert lms.running is False

    @patch("llm_benchmark.backends.detection._port_is_open")
    @patch("llm_benchmark.backends.detection.shutil.which")
    def test_port_overrides(self, mock_which, mock_port):
        from llm_benchmark.backends.detection import detect_backends

        mock_which.return_value = "/usr/bin/test"
        mock_port.return_value = True

        statuses = detect_backends(port_overrides={"llama-cpp": 9090})

        llama = next(s for s in statuses if s.name == "llama-cpp")
        assert llama.port == 9090

        # Verify _port_is_open was called with overridden port
        port_calls = {c.args[1] for c in mock_port.call_args_list}
        assert 9090 in port_calls
        assert 8080 not in port_calls


class TestAutoStartBackend:
    """Tests for auto_start_backend()."""

    @patch("llm_benchmark.backends.detection.httpx")
    @patch("llm_benchmark.backends.detection.subprocess.Popen")
    @patch("llm_benchmark.backends.detection.Path")
    def test_start_ollama(self, mock_path_cls, mock_popen, mock_httpx, tmp_path):
        from llm_benchmark.backends.detection import auto_start_backend

        # Mock Path for log directory
        mock_path_inst = MagicMock()
        mock_path_cls.return_value = mock_path_inst
        mock_path_inst.__truediv__ = lambda self, key: tmp_path / key

        # Mock the open for log file
        mock_log_file = MagicMock()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # Health check succeeds on first try
        mock_client = MagicMock()
        mock_httpx.Client.return_value.__enter__ = Mock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = Mock(return_value=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        with patch("builtins.open", mock_open()):
            result = auto_start_backend("ollama", timeout=5, log_dir=str(tmp_path))

        assert result is mock_proc
        # Verify ollama serve command was used
        popen_args = mock_popen.call_args
        assert popen_args[0][0] == ["ollama", "serve"]

    @patch("llm_benchmark.backends.detection.httpx")
    @patch("llm_benchmark.backends.detection.subprocess.Popen")
    def test_start_llama_cpp_with_model(self, mock_popen, mock_httpx, tmp_path):
        from llm_benchmark.backends.detection import auto_start_backend

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        mock_client = MagicMock()
        mock_httpx.Client.return_value.__enter__ = Mock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = Mock(return_value=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        with patch("builtins.open", mock_open()):
            result = auto_start_backend(
                "llama-cpp",
                model_path="/path/to/model.gguf",
                timeout=5,
                log_dir=str(tmp_path),
            )

        assert result is mock_proc
        popen_args = mock_popen.call_args
        cmd = popen_args[0][0]
        assert cmd[0] == "llama-server"
        assert "-m" in cmd
        assert "/path/to/model.gguf" in cmd
        assert "--port" in cmd
        assert "--host" in cmd

    def test_start_llama_cpp_without_model_raises(self):
        from llm_benchmark.backends.detection import auto_start_backend

        with pytest.raises(ValueError, match="model_path"):
            auto_start_backend("llama-cpp")

    @patch("llm_benchmark.backends.detection.httpx")
    @patch("llm_benchmark.backends.detection.subprocess.Popen")
    def test_start_lm_studio(self, mock_popen, mock_httpx, tmp_path):
        from llm_benchmark.backends.detection import auto_start_backend

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        mock_client = MagicMock()
        mock_httpx.Client.return_value.__enter__ = Mock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = Mock(return_value=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        with patch("builtins.open", mock_open()):
            result = auto_start_backend("lm-studio", timeout=5, log_dir=str(tmp_path))

        assert result is mock_proc
        popen_args = mock_popen.call_args
        assert popen_args[0][0] == ["lms", "server", "start"]

    @patch("llm_benchmark.backends.detection.time.sleep")
    @patch("llm_benchmark.backends.detection.time.monotonic")
    @patch("llm_benchmark.backends.detection.httpx")
    @patch("llm_benchmark.backends.detection.subprocess.Popen")
    def test_timeout_raises_backend_error(
        self, mock_popen, mock_httpx, mock_monotonic, mock_sleep, tmp_path
    ):
        from llm_benchmark.backends import BackendError
        from llm_benchmark.backends.detection import auto_start_backend

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # Health check always fails
        mock_client = MagicMock()
        mock_httpx.Client.return_value.__enter__ = Mock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = Mock(return_value=False)
        mock_client.get.side_effect = Exception("Connection refused")

        # Simulate time passing beyond timeout
        mock_monotonic.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        with patch("builtins.open", mock_open()):
            with pytest.raises(BackendError, match="failed to start"):
                auto_start_backend("ollama", timeout=5, log_dir=str(tmp_path))

        mock_proc.terminate.assert_called_once()

    @patch("llm_benchmark.backends.detection.httpx")
    @patch("llm_benchmark.backends.detection.subprocess.Popen")
    def test_log_file_created(self, mock_popen, mock_httpx, tmp_path):
        from llm_benchmark.backends.detection import auto_start_backend

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        mock_client = MagicMock()
        mock_httpx.Client.return_value.__enter__ = Mock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = Mock(return_value=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        m = mock_open()
        with patch("builtins.open", m):
            auto_start_backend("ollama", timeout=5, log_dir=str(tmp_path))

        # Verify log file was opened
        m.assert_called()
        log_path = str(m.call_args[0][0])
        assert "ollama.log" in log_path


class TestGetInstallInstructions:
    """Tests for get_install_instructions()."""

    def test_ollama_darwin(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("ollama", "Darwin")
        assert "curl" in result or "install" in result.lower()

    def test_ollama_linux(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("ollama", "Linux")
        assert "curl" in result or "install" in result.lower()

    def test_ollama_windows(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("ollama", "Windows")
        assert "irm" in result or "install" in result.lower()

    def test_llama_cpp_darwin(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("llama-cpp", "Darwin")
        assert "brew" in result

    def test_llama_cpp_linux(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("llama-cpp", "Linux")
        assert "apt" in result or "snap" in result

    def test_llama_cpp_windows(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("llama-cpp", "Windows")
        assert "winget" in result

    def test_lm_studio_darwin(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("lm-studio", "Darwin")
        assert "lmstudio.ai" in result

    def test_lm_studio_linux(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("lm-studio", "Linux")
        assert "lmstudio.ai" in result
        assert "No MLX" in result or "no MLX" in result.lower() or "Linux" in result

    def test_lm_studio_windows(self):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("lm-studio", "Windows")
        assert "lmstudio.ai" in result

    @patch("llm_benchmark.backends.detection.platform.system", return_value="Darwin")
    def test_default_os_detection(self, mock_platform):
        from llm_benchmark.backends.detection import get_install_instructions

        result = get_install_instructions("ollama")
        assert isinstance(result, str)
        assert len(result) > 0
        mock_platform.assert_called_once()
