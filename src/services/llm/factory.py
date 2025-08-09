"""
LLM Provider Factory
Manages different LLM providers (Gemini, OpenAI, Anthropic)
"""

from typing import Dict, Any, Optional
from src.config.settings import settings
from .providers.gemini import GeminiProvider
from .providers.openai_ import OpenAIProvider
from .providers.anthropic_ import AnthropicProvider


class LLMFactory:
    """Factory for creating LLM providers"""
    
    _providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    
    @classmethod
    def create_provider(cls, provider: Optional[str] = None, **kwargs) -> Any:
        """
        Create LLM provider instance
        
        Args:
            provider: Provider name (gemini, openai, anthropic)
            **kwargs: Additional provider parameters
            
        Returns:
            LLM provider instance
        """
        provider_name = (provider or settings.llm_provider).lower()
        
        if provider_name not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider_name}. Available: {list(cls._providers.keys())}")
        
        provider_class = cls._providers[provider_name]
        
        # Get API key based on provider
        api_keys = {
            "gemini": settings.gemini_api_key,
            "openai": settings.openai_api_key,
            "anthropic": settings.anthropic_api_key,
        }
        
        api_key = api_keys.get(provider_name)
        if not api_key:
            raise ValueError(f"API key not found for provider: {provider_name}")
        
        return provider_class(
            model=settings.llm_model,
            api_key=api_key,
            **kwargs
        )
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, Any]:
        """Get list of available providers"""
        return {
            name: {
                "name": name,
                "available": cls._is_provider_available(name)
            }
            for name in cls._providers.keys()
        }
    
    @classmethod
    def _is_provider_available(cls, provider: str) -> bool:
        """Check if provider is available (API keys set)"""
        if provider == "gemini":
            return bool(settings.gemini_api_key)
        elif provider == "openai":
            return bool(settings.openai_api_key)
        elif provider == "anthropic":
            return bool(settings.anthropic_api_key)
        return False
