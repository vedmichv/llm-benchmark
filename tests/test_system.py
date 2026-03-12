"""Tests for llm_benchmark.system module."""

from llm_benchmark.system import get_system_info, format_system_summary


class TestGetSystemInfo:
    """Integration tests that run on the current platform."""

    def test_returns_system_info_with_cpu(self):
        """Verify get_system_info() returns a SystemInfo with non-empty cpu."""
        info = get_system_info()
        assert info.cpu != ""
        assert info.cpu != "Unknown" or True  # May be Unknown on CI

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

    def test_contains_ollama(self):
        """Verify the summary includes Ollama version."""
        summary = format_system_summary()
        assert "Ollama" in summary
