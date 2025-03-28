"""Tests for the PromptLoader utility."""

import os

import pytest

from prompt_tuner.utils.prompt_loader import PromptLoader


def test_prompt_loader_initialization():
    """Test that PromptLoader initializes correctly."""
    loader = PromptLoader()
    assert loader is not None
    assert hasattr(loader, "prompt_dir")
    assert hasattr(loader, "cache")
    assert isinstance(loader.cache, dict)


def test_format_with_variables():
    """Test formatting a prompt with variables."""
    # Create a test prompt file
    os.makedirs("test_prompts", exist_ok=True)
    with open("test_prompts/test.prompt", "w") as f:
        f.write("Hello, {name}! The answer is {answer}.")

    try:
        loader = PromptLoader("test_prompts")
        result = loader.format("test", name="World", answer=42)
        assert result == "Hello, World! The answer is 42."
    finally:
        # Clean up
        os.remove("test_prompts/test.prompt")
        os.rmdir("test_prompts")


def test_cache_usage():
    """Test that caching works as expected."""
    os.makedirs("test_prompts", exist_ok=True)
    with open("test_prompts/cache_test.prompt", "w") as f:
        f.write("Original content")

    try:
        loader = PromptLoader("test_prompts")

        # Initial load should cache the content
        first_load = loader.load("cache_test")
        assert first_load == "Original content"
        assert "cache_test" in loader.cache

        # Change the file content
        with open("test_prompts/cache_test.prompt", "w") as f:
            f.write("Updated content")

        # With cache, should still get original content
        cached_load = loader.load("cache_test")
        assert cached_load == "Original content"

        # Without cache, should get updated content
        uncached_load = loader.load("cache_test", use_cache=False)
        assert uncached_load == "Updated content"

        # Cache should be updated now
        assert loader.cache["cache_test"] == "Updated content"

        # Clear cache and check
        loader.clear_cache()
        assert len(loader.cache) == 0
    finally:
        # Clean up
        os.remove("test_prompts/cache_test.prompt")
        os.rmdir("test_prompts")


def test_file_not_found():
    """Test that appropriate error is raised when file not found."""
    loader = PromptLoader("nonexistent_dir")
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent_prompt")
