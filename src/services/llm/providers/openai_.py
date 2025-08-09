from __future__ import annotations
"""OpenAI LLM provider"""

import httpx
from typing import Dict, Any, Optional

from src.config.settings import settings
from src.services.llm.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key or settings.openai_api_key, **kwargs)
        self.base_url = "https://api.openai.com/v1"
        self.model_name = model
        
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = []
        
        # Add system prompt if provided
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 0.95),
        }
        
        url = f"{self.base_url}/chat/completions"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract text and usage
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            
            return LLMResponse(
                content=text,
                usage=usage,
                model=self.model_name,
                provider="openai"
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model_name,
            "max_tokens": 4096,
            "supports_system_prompt": True
        }
