"""Tests for llm_benchmark.system module."""

from unittest.mock import patch

from llm_benchmark.system import (
    format_system_summary,
    get_backend_inventory,
    get_system_info,
)


class TestGetSystemInfo:
    """Integration tests that run on the current platform."""

    def test_returns_system_info_with_cpu(self):
        """Verify get_system_info() returns a SystemInfo with non-empty cpu."""
        info = get_system_info()
        assert info.cpu != ""
        # CPU may be "Unknown" on CI, so we only check it's non-empty
        assert isinstance(info.cpu, str)

    def test_returns_positive_ram(self):
        """Verify ram_gb > 0 on the current platform."""
        info = get_system_info()
        assert info.ram_gb > 0

    def test_returns_os_name(self):
        """Verify os_name is non-empty."""
        info = get_system_info()
        assert info.os_name != ""

    def test_returns_python_version(self):
        """Verify python_version looks like a version string."""
        info = get_system_info()
        assert "." in info.python_version

    def test_returns_backend_fields(self):
        """Verify backend_name and backend_version are populated."""
        info = get_system_info()
        assert info.backend_name == "ollama"
        assert isinstance(info.backend_version, str)


class TestFormatSystemSummary:
    """Test the compact one-line summary formatter."""

    def test_returns_nonempty_string(self):
        """Verify format_system_summary() returns a non-empty string."""
        summary = format_system_summary()
        assert len(summary) > 0

    def test_contains_ram(self):
        """Verify the summary includes RAM info."""
        summary = format_system_summary()
        assert "GB RAM" in summary

    def test_contains_backend(self):
        """Verify the summary includes backend version."""
        summary = format_system_summary()
        assert "Ollama" in summary


def _mock_backend_status(name, installed, running, port):
    """Create a mock BackendStatus."""
    from llm_benchmark.backends.detection import BackendStatus

    return BackendStatus(
        name=name,
        installed=installed,
        running=running,
        binary_path=f"/usr/bin/{name}" if installed else None,
        port=port,
    )


class TestFormatSystemSummaryBackends:
    """Test that format_system_summary includes backend detection section."""

    def test_shows_running_backend(self):
        """Running backends appear with 'running' and port."""
        statuses = [
            _mock_backend_status("ollama", True, True, 11434),
            _mock_backend_status("llama-cpp", False, False, 8080),
            _mock_backend_status("lm-studio", False, False, 1234),
        ]
        with patch(
            "llm_benchmark.backends.detection.detect_backends",
            return_value=statuses,
        ):
            summary = format_system_summary()
            assert "running" in summary
            assert "11434" in summary

    def test_shows_installed_not_running(self):
        """Installed-but-not-running backends appear with yellow status."""
        statuses = [
            _mock_backend_status("ollama", True, True, 11434),
            _mock_backend_status("llama-cpp", True, False, 8080),
            _mock_backend_status("lm-studio", False, False, 1234),
        ]
        with patch(
            "llm_benchmark.backends.detection.detect_backends",
            return_value=statuses,
        ):
            summary = format_system_summary()
            assert "installed, not running" in summary

    def test_shows_not_installed(self):
        """Not-installed backends appear with dim styling."""
        statuses = [
            _mock_backend_status("ollama", True, True, 11434),
            _mock_backend_status("llama-cpp", False, False, 8080),
            _mock_backend_status("lm-studio", False, False, 1234),
        ]
        with patch(
            "llm_benchmark.backends.detection.detect_backends",
            return_value=statuses,
        ):
            summary = format_system_summary()
            assert "not installed" in summary

    def test_shows_active_backend(self):
        """Active backend shown prominently."""
        statuses = [
            _mock_backend_status("ollama", True, True, 11434),
        ]
        with patch(
            "llm_benchmark.backends.detection.detect_backends",
            return_value=statuses,
        ):
            summary = format_system_summary()
            assert "Active backend:" in summary


class TestGetBackendInventory:
    """Test the detailed backend inventory for the info command."""

    def test_all_installed_and_running(self):
        """Inventory shows all backends when all are installed and running."""
        statuses = [
            _mock_backend_status("ollama", True, True, 11434),
            _mock_backend_status("llama-cpp", True, True, 8080),
            _mock_backend_status("lm-studio", True, True, 1234),
        ]
        with patch(
            "llm_benchmark.backends.detection.detect_backends",
            return_value=statuses,
        ):
            inventory = get_backend_inventory()
            assert "ollama" in inventory
            assert "llama-cpp" in inventory
            assert "lm-studio" in inventory
            assert "Backend Inventory" in inventory

    def test_partial_installation(self):
        """Inventory shows install instructions for missing backends."""
        statuses = [
            _mock_backend_status("ollama", True, True, 11434),
            _mock_backend_status("llama-cpp", False, False, 8080),
            _mock_backend_status("lm-studio", False, False, 1234),
        ]
        with patch(
            "llm_benchmark.backends.detection.detect_backends",
            return_value=statuses,
        ):
            inventory = get_backend_inventory()
            assert "install:" in inventory
            assert "installed=False" in inventory

    def test_none_installed(self):
        """Inventory handles the case where no backends are installed."""
        statuses = [
            _mock_backend_status("ollama", False, False, 11434),
            _mock_backend_status("llama-cpp", False, False, 8080),
            _mock_backend_status("lm-studio", False, False, 1234),
        ]
        with patch(
            "llm_benchmark.backends.detection.detect_backends",
            return_value=statuses,
        ):
            inventory = get_backend_inventory()
            assert inventory.count("installed=False") == 3
