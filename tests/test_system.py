"""Tests for llm_benchmark.system module."""

from llm_benchmark.system import format_system_summary, get_system_info


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
