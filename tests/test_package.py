"""Tests for package import and structure validation."""


def test_imports():
    """Verify that the package can be imported and has a version string."""
    import llm_benchmark

    assert hasattr(llm_benchmark, "__version__")
    assert isinstance(llm_benchmark.__version__, str)
    assert len(llm_benchmark.__version__) > 0


def test_structure():
    """Verify all expected submodules are importable (QUAL-02)."""
    import importlib

    # models.py is created in Task 2; test config and prompts for now
    for module_name in ["llm_benchmark.config", "llm_benchmark.prompts"]:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"
