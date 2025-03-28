"""
Type stubs for lmstudio_client
"""
from typing import Any, Dict, List, Optional, AsyncIterator

class LMStudioClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
    ) -> None: ...
    
    async def list_models(self) -> Dict[str, Any]: ...
    
    async def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]: ...
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]: ...
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]: ...