# FILE: backend/providers/grok.py
"""
Grok (xAI) provider adapter
"""
import logging
import httpx
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GrokProvider:
    """Grok provider (xAI)"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1"
        logger.info(f"Grok provider: model={model}")
    
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate text using Grok"""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = httpx.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        
        return {
            "text": text,
            "model": self.model,
            "usage": data.get("usage", {})
        }
