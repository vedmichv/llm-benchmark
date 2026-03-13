"""Tests for model recommendation tiers and filtering."""

from __future__ import annotations

from llm_benchmark.recommend import (
    filter_already_installed,
    get_recommended_models,
)


class TestGetRecommendedModels:
    """Verify RAM-based tier filtering."""

    def test_8gb_returns_only_small_tier(self):
        models = get_recommended_models(ram_gb=8.0)
        tiers = {m["tier"] for m in models}
        assert tiers == {"small"}
        assert len(models) > 0

    def test_16gb_returns_small_and_medium(self):
        models = get_recommended_models(ram_gb=16.0)
        tiers = {m["tier"] for m in models}
        assert tiers == {"small", "medium"}

    def test_36gb_returns_small_medium_large(self):
        models = get_recommended_models(ram_gb=36.0)
        tiers = {m["tier"] for m in models}
        assert tiers == {"small", "medium", "large"}

    def test_64gb_returns_all_tiers(self):
        models = get_recommended_models(ram_gb=64.0)
        tiers = {m["tier"] for m in models}
        assert tiers == {"small", "medium", "large", "xl"}

    def test_each_model_has_required_keys(self):
        models = get_recommended_models(ram_gb=64.0)
        for m in models:
            assert "name" in m
            assert "size_label" in m
            assert "tier" in m
            assert "description" in m

    def test_low_ram_still_returns_something(self):
        models = get_recommended_models(ram_gb=4.0)
        assert len(models) > 0
        assert all(m["tier"] == "small" for m in models)


class TestFilterAlreadyInstalled:
    """Verify installed model filtering."""

    def test_no_overlap_returns_full_list(self):
        recommended = [
            {"name": "llama3.2:1b", "tier": "small", "size_label": "1B", "description": "Fast"},
            {"name": "phi4-mini", "tier": "small", "size_label": "3.8B", "description": "Compact"},
        ]
        result = filter_already_installed(recommended, ["gemma3:4b"])
        assert len(result) == 2

    def test_all_overlap_returns_empty(self):
        recommended = [
            {"name": "llama3.2:1b", "tier": "small", "size_label": "1B", "description": "Fast"},
        ]
        result = filter_already_installed(recommended, ["llama3.2:1b"])
        assert len(result) == 0

    def test_partial_overlap_removes_installed(self):
        recommended = [
            {"name": "llama3.2:1b", "tier": "small", "size_label": "1B", "description": "Fast"},
            {"name": "phi4-mini", "tier": "small", "size_label": "3.8B", "description": "Compact"},
            {"name": "qwen2.5:0.5b", "tier": "small", "size_label": "0.5B", "description": "Tiny"},
        ]
        result = filter_already_installed(recommended, ["phi4-mini"])
        assert len(result) == 2
        names = [m["name"] for m in result]
        assert "phi4-mini" not in names

    def test_empty_installed_returns_all(self):
        recommended = [
            {"name": "llama3.2:1b", "tier": "small", "size_label": "1B", "description": "Fast"},
        ]
        result = filter_already_installed(recommended, [])
        assert len(result) == 1
