# FILE: backend/providers/lmstudio.py
"""
LMStudio provider adapter
"""
import logging
import httpx
from typing import Dict, Any

from backend.providers.offline import OfflineProvider

logger = logging.getLogger(__name__)


class LMStudioProvider(OfflineProvider):
    """LMStudio provider using OpenAI-compatible API"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        logger.info(f"LMStudio provider: {base_url}, model: {model}")
    
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate text using LMStudio"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = httpx.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        
        return {
            "text": text,
            "model": self.model,
            "usage": data.get("usage", {})
        }
