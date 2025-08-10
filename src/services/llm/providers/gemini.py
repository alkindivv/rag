from __future__ import annotations
"""Gemini LLM provider for Flash 2.0"""

import json
from typing import Dict, Any, Optional
import httpx

from src.config.settings import settings
from src.services.llm.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Gemini Flash 2.0 provider implementation"""
    
    def __init__(self, model: str = "gemini-2.0-flash-lite", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key or settings.gemini_api_key, **kwargs)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model_name = model
        
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Gemini API"""
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.3),
                "maxOutputTokens": kwargs.get("max_tokens", 1000),
                "topP": kwargs.get("top_p", 0.95),
            }
        }
        
        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract text from response
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            
            usage = data.get("usageMetadata", {})
            
            return LLMResponse(
                content=text,
                usage=usage,
                model=self.model_name,
                provider="gemini"
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model_name,
            "max_tokens": 8192,
            "supports_system_prompt": False
        }
