# FILE: backend/providers/registry.py (Updated with I/O capture)

"""
Provider registry with I/O capture for debugging and transparency
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from backend.config import get_settings
from backend.providers.lmstudio import LMStudioProvider
from backend.providers.ollama import OllamaProvider
from backend.providers.openai import OpenAIProvider
from backend.providers.openrouter import OpenRouterProvider
from backend.providers.hf import HuggingFaceProvider
from backend.providers.grok import GrokProvider
from backend.providers.gemini import GeminiProvider
from backend.governance.redaction import redact_pii

logger = logging.getLogger(__name__)
settings = get_settings()


class CircuitBreaker:
    """Simple circuit breaker for provider failover"""
    
    def __init__(self, threshold: int = 3, timeout_seconds: int = 60):
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self.failures = {}
        self.open_until = {}
    
    def record_failure(self, provider: str):
        """Record a failure for provider"""
        self.failures[provider] = self.failures.get(provider, 0) + 1
        if self.failures[provider] >= self.threshold:
            self.open_until[provider] = datetime.utcnow() + timedelta(seconds=self.timeout_seconds)
            logger.warning(f"Circuit breaker opened for {provider}")
    
    def record_success(self, provider: str):
        """Record a success for provider"""
        self.failures[provider] = 0
        if provider in self.open_until:
            del self.open_until[provider]
    
    def is_open(self, provider: str) -> bool:
        """Check if circuit is open for provider"""
        if provider in self.open_until:
            if datetime.utcnow() < self.open_until[provider]:
                return True
            else:
                # Timeout expired, reset
                del self.open_until[provider]
                self.failures[provider] = 0
        return False


class ProviderRegistry:
    """Registry of LLM providers with routing logic and I/O capture"""
    
    def __init__(self):
        self.providers = {}
        self.circuit_breaker = CircuitBreaker()
        self.io_log = []  # Store recent I/O for debugging
        self.max_io_log_size = 100
        self._initialize_providers()
        self._setup_io_log_dir()
    
    def _setup_io_log_dir(self):
        """Setup I/O log directory"""
        self.io_log_dir = Path(settings.logs_dir) / "provider_io"
        self.io_log_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        # Offline providers
        try:
            self.providers["lmstudio"] = LMStudioProvider(
                base_url=settings.lmstudio_base_url,
                model=settings.lmstudio_model
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LMStudio: {e}")
        
        try:
            self.providers["ollama"] = OllamaProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")
        
        # Online providers
        if settings.openai_api_key:
            try:
                self.providers["openai"] = OpenAIProvider(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        if settings.openrouter_api_key:
            try:
                self.providers["openrouter"] = OpenRouterProvider(
                    api_key=settings.openrouter_api_key,
                    model=settings.openrouter_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter: {e}")
        
        if settings.huggingface_api_key:
            try:
                self.providers["huggingface"] = HuggingFaceProvider(
                    api_key=settings.huggingface_api_key,
                    model=settings.huggingface_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace: {e}")
        
        if settings.grok_api_key:
            try:
                self.providers["grok"] = GrokProvider(
                    api_key=settings.grok_api_key,
                    model=settings.grok_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Grok: {e}")
        
        if settings.gemini_api_key:
            try:
                self.providers["gemini"] = GeminiProvider(
                    api_key=settings.gemini_api_key,
                    model=settings.gemini_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        logger.info(f"Initialized providers: {list(self.providers.keys())}")
    
    def _get_provider_order(self) -> List[str]:
        """Get provider order based on policy"""
        if settings.router_policy == "offline_first":
            return ["lmstudio", "ollama", "openrouter", "openai", "huggingface", "grok", "gemini"]
        elif settings.router_policy == "online_first":
            return ["openai", "openrouter", "huggingface", "gemini", "grok", "lmstudio", "ollama"]
        else:  # round_robin
            return list(self.providers.keys())
    
    def _sanitize_prompt(self, prompt: str, max_length: int = 2000) -> str:
        """Sanitize prompt for display (truncate + redact PII)"""
        if settings.redaction_enabled:
            prompt = redact_pii(prompt)
        
        if len(prompt) > max_length:
            return prompt[:max_length] + f"\n\n[... truncated {len(prompt) - max_length} chars]"
        
        return prompt
    
    def _log_io(
        self,
        provider: str,
        model: str,
        prompt: str,
        output: str,
        correlation_id: Optional[str],
        duration_ms: int,
        error: Optional[str] = None
    ):
        """Log provider I/O for debugging"""
        io_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "provider": provider,
            "model": model,
            "prompt_length": len(prompt),
            "prompt": prompt,
            "prompt_sanitized": self._sanitize_prompt(prompt),
            "output_length": len(output) if output else 0,
            # "output": output[:1000] if output else None,  # First 1000 chars
            "output": output,
            "duration_ms": duration_ms,
            "error": error
        }
        
        # Add to in-memory log (ring buffer)
        self.io_log.append(io_entry)
        if len(self.io_log) > self.max_io_log_size:
            self.io_log = self.io_log[-self.max_io_log_size:]
        
        # Persist to disk (daily log file)
        log_file = self.io_log_dir / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(io_entry, ensure_ascii=False) + '\n')
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text using first available provider with I/O capture
        
        Returns:
            {text: str, provider: str, model: str, metadata: dict}
        """
        provider_order = self._get_provider_order()
        start_time = datetime.utcnow()
        
        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue
            
            if self.circuit_breaker.is_open(provider_name):
                logger.debug(f"Skipping {provider_name} (circuit open)")
                continue
            
            provider = self.providers[provider_name]
            provider_start = datetime.utcnow()
            
            try:
                logger.info(f"[{correlation_id}] Trying provider: {provider_name}")
                result = provider.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                duration_ms = int((datetime.utcnow() - provider_start).total_seconds() * 1000)
                
                self.circuit_breaker.record_success(provider_name)
                
                # Log I/O for debugging
                self._log_io(
                    provider=provider_name,
                    model=result.get("model", "unknown"),
                    prompt=prompt,
                    output=result["text"],
                    correlation_id=correlation_id,
                    duration_ms=duration_ms
                )
                
                return {
                    **result,
                    "provider": provider_name,
                    "router_reason": f"Selected by policy: {settings.router_policy}",
                    "duration_ms": duration_ms
                }
            
            except Exception as e:
                duration_ms = int((datetime.utcnow() - provider_start).total_seconds() * 1000)
                
                logger.warning(f"[{correlation_id}] Provider {provider_name} failed: {e}")
                self.circuit_breaker.record_failure(provider_name)
                
                # Log failed attempt
                self._log_io(
                    provider=provider_name,
                    model="unknown",
                    prompt=prompt,
                    output="",
                    correlation_id=correlation_id,
                    duration_ms=duration_ms,
                    error=str(e)
                )
                
                if not settings.router_fallback:
                    raise
                # Continue to next provider
        
        # No provider succeeded
        raise RuntimeError("All LLM providers failed")
    
    def get_recent_io(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent I/O entries"""
        return self.io_log[-limit:]
    
    def clear_io_log(self):
        """Clear in-memory I/O log"""
        self.io_log = []


# Global registry singleton
_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get or create global provider registry"""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry

