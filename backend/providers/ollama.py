# FILE: backend/providers/ollama.py
"""
Ollama provider adapter
"""
import logging
import httpx
from typing import Dict, Any

from backend.providers.offline import OfflineProvider

logger = logging.getLogger(__name__)


class OllamaProvider(OfflineProvider):
    """Ollama provider"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        logger.info(f"Ollama provider: {base_url}, model: {model}")
    
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate text using Ollama"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }
        
        response = httpx.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        text = data["response"]
        
        return {
            "text": text,
            "model": self.model,
            "usage": {}
        }
