"""
Prompt loading utilities for Prompt Tuner.
"""

import os
from typing import Any, Dict, Optional


class PromptLoader:
    """Loads prompt templates from files."""

    def __init__(self, prompt_dir: Optional[str] = None) -> None:
        """
        Initialize the prompt loader.

        Args:
            prompt_dir: Directory containing prompt template files
        """
        if prompt_dir is None:
            # Default to prompt_tuner/prompts directory
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.prompt_dir = os.path.join(package_dir, "prompts")
        else:
            self.prompt_dir = prompt_dir

        self.cache: Dict[str, str] = {}

    def load(self, name: str, use_cache: bool = True) -> str:
        """
        Load a prompt template from a file.

        Args:
            name: Name of the prompt template file (without extension)
            use_cache: Whether to use cached templates

        Returns:
            The prompt template as a string
        """
        if use_cache and name in self.cache:
            return self.cache[name]

        file_path = os.path.join(self.prompt_dir, f"{name}.prompt")

        if not os.path.exists(file_path):
            # Try alternate locations
            alt_paths = [
                os.path.join("prompts", f"{name}.prompt"),  # Local prompts dir
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "prompts",
                    f"{name}.prompt",
                ),  # Package prompts dir
            ]

            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
            else:
                # If we get here, no file was found
                raise FileNotFoundError(
                    f"Prompt template not found: {file_path}. Also tried: {alt_paths}"
                )

        with open(file_path, "r") as f:
            template = f.read()

        # Always update the cache with the latest content, regardless of use_cache
        # This ensures the cache is up-to-date after any file read
        self.cache[name] = template

        return template

    def format(self, name: str, **kwargs: Any) -> str:
        """
        Load and format a prompt template with variables.

        Args:
            name: Name of the prompt template file (without extension)
            **kwargs: Variables to substitute in the template

        Returns:
            The formatted prompt string
        """
        template = self.load(name)
        return template.format(**kwargs)

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self.cache.clear()
