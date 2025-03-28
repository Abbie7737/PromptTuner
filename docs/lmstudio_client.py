"""
LMStudio API Client

LMSTUDIO_URL=http://192.168.0.98:1234/api/v0
LMSTUDIO_SMALL_MODEL=llama-3.2-3b-instruct

"""

import json
import os
import sys
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    NotRequired,
    Optional,
    TypedDict,
)

import httpx


class LMStudioClient:
    """Client for interacting with the LMStudio REST API"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        """
        Initialize the LMStudio API client

        Args:
            base_url: The base URL of the LMStudio API
            model: The model ID to use for completions
            api_key: API key (not required for local LMStudio)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def list_models(self) -> Dict[str, Any]:
        """
        Get a list of available models

        Returns:
            Dict containing the API response with model data
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/models", headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    async def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model

        Args:
            model_id: The ID of the model to get info for, defaults to client model

        Returns:
            Dict containing the API response with model info
        """
        model_id = model_id or self.model
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/models/{model_id}", headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors
            error_message = (
                f"Failed to get model info: HTTP error {e.response.status_code}"
            )
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_message = f"API error: {error_data['error']}"
            except:
                pass
            raise Exception(error_message) from e
        except httpx.RequestError as e:
            # Handle request errors (connection problems, timeouts, etc.)
            raise Exception(f"Request error: {str(e)}") from e

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Get a chat completion from the model

        Args:
            messages: List of message objects with role and content
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate (-1 for model default)
            stream: Whether to stream the response
            tools: Optional list of tool definitions

        Returns:
            Dict containing the API response with completion data
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            # Add tool_choice to encourage tool usage
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors
            error_message = f"HTTP error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_message = f"API error: {error_data['error']}"
            except:
                pass
            raise Exception(error_message) from e
        except httpx.RequestError as e:
            # Handle request errors (connection problems, timeouts, etc.)
            raise Exception(f"Request error: {str(e)}") from e

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a chat completion from the model

        Args:
            messages: List of message objects with role and content
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate (-1 for model default)
            tools: Optional list of tool definitions

        Yields:
            Dict containing a chunk of the streaming response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,  # Always stream
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            # Add tool_choice to encourage tool usage
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    # Process the response as a stream
                    buffer = ""
                    complete_response = {}

                    async for line in response.aiter_lines():
                        # Skip empty lines
                        if not line.strip():
                            continue

                        # Handle SSE format
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix

                        # End of stream marker
                        if line == "[DONE]":
                            # If we have a complete response, yield it one last time with stats
                            if complete_response:
                                # Get non-streaming metrics to add to the final chunk
                                # Make a copy of the payload without stream=True
                                non_stream_payload = {**payload, "stream": False}

                                try:
                                    # Make a non-streaming request to get metrics
                                    async with httpx.AsyncClient(
                                        timeout=self.timeout
                                    ) as metrics_client:
                                        stats_response = await metrics_client.post(
                                            f"{self.base_url}/chat/completions",
                                            headers=self._get_headers(),
                                            json=non_stream_payload,
                                        )
                                        stats_response.raise_for_status()
                                        stats_data = stats_response.json()

                                    # Add stats to the complete_response
                                    if "stats" in stats_data:
                                        complete_response["stats"] = stats_data["stats"]
                                    if "usage" in stats_data:
                                        complete_response["usage"] = stats_data["usage"]
                                    if "model_info" in stats_data:
                                        complete_response["model_info"] = stats_data[
                                            "model_info"
                                        ]

                                    # Mark this as a final response with metrics
                                    complete_response["is_final_with_metrics"] = True

                                    # Yield the enhanced response
                                    yield complete_response
                                except Exception as e:
                                    # If getting metrics fails, yield what we have
                                    complete_response["metrics_error"] = str(e)
                                    yield complete_response
                            break

                        # Try to parse JSON
                        try:
                            # Debug output to see what we're receiving
                            if os.getenv("DEBUG_MODE"):
                                print(f"\nDEBUG RAW LINE: {line}", file=sys.stderr)

                            chunk = json.loads(line)
                            # Update our complete response with the latest chunk
                            complete_response = chunk
                            yield chunk
                        except json.JSONDecodeError as e:
                            # Debug output for JSON errors
                            if os.getenv("DEBUG_MODE"):
                                print(
                                    f"\nDEBUG JSON ERROR: {str(e)} on line: {line}",
                                    file=sys.stderr,
                                )

                            # Collect fragmented JSON (can happen in some implementations)
                            buffer += line
                            try:
                                chunk = json.loads(buffer)
                                buffer = ""  # Reset buffer after successful parse
                                complete_response = chunk
                                yield chunk
                            except json.JSONDecodeError:
                                # Still not valid JSON, keep collecting
                                pass
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors
            error_message = f"HTTP error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_message = f"API error: {error_data['error']}"
            except:
                pass
            raise Exception(error_message) from e
        except httpx.RequestError as e:
            # Handle request errors (connection problems, timeouts, etc.)
            raise Exception(f"Request error: {str(e)}") from e

    def _get_headers(self) -> Dict[str, str]:
        """Get the HTTP headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class ServiceError(Exception):
    """Exception raised for errors in the service API"""

    pass


class ServiceClientOptions(TypedDict):
    """Options for the ServiceClient"""

    base_url: str
    api_key: NotRequired[Optional[str]]
    timeout: NotRequired[int]


class ServiceClient:
    """Client for interacting with the LMStudio Services API"""

    def __init__(self, options: ServiceClientOptions) -> None:
        """
        Initialize the Services API client

        Args:
            options: Configuration options for the client
        """
        self.base_url = options["base_url"]
        self.api_key = options.get("api_key")
        self.timeout = options.get("timeout", 30)

    async def list_directory(
        self, path: str = "/", include_hidden: bool = False, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        List contents of a directory

        Args:
            path: Directory path relative to allowed base directory
            include_hidden: Whether to include hidden files starting with '.'
            use_cache: Whether to use cached results if available

        Returns:
            Dict containing the API response with directory listing
        """
        params = {
            "path": path,
            "include_hidden": str(include_hidden).lower(),
            "use_cache": str(use_cache).lower(),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/list_directory",
                    headers=self._get_headers(),
                    params=params,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            error_message = f"HTTP error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_message = f"API error: {error_data['error']}"
            except:
                pass
            raise ServiceError(error_message) from e
        except httpx.RequestError as e:
            raise ServiceError(f"Request error: {str(e)}") from e

    def _get_headers(self) -> Dict[str, str]:
        """Get the HTTP headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
