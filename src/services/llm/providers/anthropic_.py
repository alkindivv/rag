from __future__ import annotations
"""Anthropic Claude provider"""

import httpx
from typing import Dict, Any, Optional

from src.config.settings import settings
from src.services.llm.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key or settings.anthropic_api_key, **kwargs)
        self.base_url = "https://api.anthropic.com/v1"
        self.model_name = model
        
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.3),
        }
        
        # Add system prompt if provided
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            payload["system"] = system_prompt
        
        url = f"{self.base_url}/messages"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract text and usage
            text = data.get("content", [{}])[0].get("text", "")
            usage = data.get("usage", {})
            
            return LLMResponse(
                content=text,
                usage=usage,
                model=self.model_name,
                provider="anthropic"
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "anthropic",
            "model": self.model_name,
            "max_tokens": 4096,
            "supports_system_prompt": True
        }
