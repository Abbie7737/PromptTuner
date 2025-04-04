"""
LMStudio client utilities for Prompt Tuner.
"""

import os

# Import the LMStudio client from the docs folder
import sys
from typing import Any, Dict, List

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")
)
from lmstudio_client import LMStudioClient


class LMStudioManager:
    """Manages LMStudio client instances for large and small models."""

    def __init__(self, env_file: str = ".env") -> None:
        """
        Initialize the LMStudio manager.

        Args:
            env_file: Path to the .env file with configuration
        """
        self._load_env(env_file)
        self.large_model = self._create_large_model_client()
        self.small_model = self._create_small_model_client()

    def _load_env(self, env_file: str) -> None:
        """Load environment variables from the .env file."""
        if not os.path.exists(env_file):
            raise FileNotFoundError(f"Environment file not found: {env_file}")

        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip("\"'")

    def _create_large_model_client(self) -> LMStudioClient:
        """Create a client for the large model."""
        # Get timeout with a high default for reasoning models
        timeout = int(os.environ.get("LMSTUDIO_TIMEOUT", "300"))
        print(f"Large model timeout set to {timeout} seconds")

        return LMStudioClient(
            base_url=os.environ.get("LMSTUDIO_URL", "http://localhost:1234/api/v0"),
            model=os.environ.get("LMSTUDIO_LARGE_MODEL", "mistral-7b-instruct"),
            timeout=timeout,
        )

    def _create_small_model_client(self) -> LMStudioClient:
        """Create a client for the small model."""
        # Get timeout with a reasonable default
        timeout = int(os.environ.get("LMSTUDIO_TIMEOUT", "300"))
        print(f"Small model timeout set to {timeout} seconds")

        return LMStudioClient(
            base_url=os.environ.get("LMSTUDIO_URL", "http://localhost:1234/api/v0"),
            model=os.environ.get("LMSTUDIO_SMALL_MODEL", "phi-2"),
            timeout=timeout,
        )

    async def run_on_large_model(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """
        Run a prompt on the large model.

        Args:
            messages: List of message objects with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            API response with completion data

        Raises:
            ValueError: When there's a model-specific error that can't be recovered
        """
        model_name = os.environ.get("LMSTUDIO_LARGE_MODEL", "mistral-7b-instruct")
        
        try:
            result = await self.large_model.chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return dict(result)
        except Exception as e:
            error_msg = str(e)
            
            # Check for model-specific errors
            if "Cannot perform inplace addition on an mlx.core.array and str" in error_msg:
                raise ValueError(
                    f"Error connecting to model '{model_name}': MLX array error detected. "
                    f"This is typically resolved by restarting LM Studio. "
                    f"If the issue persists after restart, try a different model."
                )
            
            # Check for common recoverable errors
            if "400 Bad Request" in error_msg:
                print(f"Model '{model_name}' returned a bad request error - attempting with reduced message content")

                # Simplify messages by truncating if they're too long
                simplified_messages = []
                for msg in messages:
                    if len(msg["content"]) > 8000:  # If content is very long, truncate it
                        msg["content"] = msg["content"][:8000] + "... [truncated]"
                    simplified_messages.append(msg)

                try:
                    # Try again with simplified messages
                    result = await self.large_model.chat_completion(
                        messages=simplified_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return dict(result)
                except Exception as retry_error:
                    # If retry fails, provide a more helpful error message
                    raise ValueError(
                        f"The model '{model_name}' could not process the request, even after simplification. "
                        f"Error: {str(retry_error)}. Please restart LM Studio and try again."
                    ) from retry_error
            else:
                # For other errors, provide a more helpful error message
                raise ValueError(
                    f"Error with model '{model_name}': {error_msg}. "
                    f"Please restart LM Studio, which resolves most issues. "
                    f"If the problem persists after restart, try a different model."
                ) from e

    async def run_on_small_model(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Run a prompt on the small model.

        Args:
            messages: List of message objects with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            API response with completion data
            
        Raises:
            ValueError: When there's a model-specific error that can't be recovered
        """
        model_name = os.environ.get("LMSTUDIO_SMALL_MODEL", "phi-2")
        
        try:
            result = await self.small_model.chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return dict(result)
        except Exception as e:
            error_msg = str(e)
            
            # Check for model-specific errors
            if "Cannot perform inplace addition on an mlx.core.array and str" in error_msg:
                raise ValueError(
                    f"Error connecting to model '{model_name}': MLX array error detected. "
                    f"This is typically resolved by restarting LM Studio. "
                    f"If the issue persists after restart, try a different model."
                )
            
            # Check for common recoverable errors
            if "400 Bad Request" in error_msg or "ReadTimeout" in error_msg:
                print(f"Model '{model_name}' returned an error - attempting with reduced message content")

                # Simplify messages by truncating if they're too long
                simplified_messages = []
                for msg in messages:
                    if len(msg["content"]) > 4000:  # Small models have less context
                        msg["content"] = msg["content"][:4000] + "... [truncated]"
                    simplified_messages.append(msg)

                try:
                    # Try again with simplified messages
                    result = await self.small_model.chat_completion(
                        messages=simplified_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return dict(result)
                except Exception as retry_error:
                    # If retry fails, provide a more helpful error message
                    raise ValueError(
                        f"The model '{model_name}' could not process the request, even after simplification. "
                        f"Error: {str(retry_error)}. Please restart LM Studio and try again."
                    ) from retry_error
            else:
                # For other errors, provide a more helpful error message
                raise ValueError(
                    f"Error with model '{model_name}': {error_msg}. "
                    f"Please restart LM Studio, which resolves most issues. "
                    f"If the problem persists after restart, try a different model."
                ) from e
