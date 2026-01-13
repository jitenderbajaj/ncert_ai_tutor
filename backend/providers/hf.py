# FILE: backend/providers/hf.py
"""
Hugging Face Inference API provider adapter
"""
import logging
import httpx
from typing import Dict, Any

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """Hugging Face Inference API provider"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api-inference.huggingface.co"
        logger.info(f"HuggingFace provider: model={model}")
    
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate text using Hugging Face"""
        url = f"{self.base_url}/models/{self.model}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        response = httpx.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if isinstance(data, list):
            text = data[0]["generated_text"]
        else:
            text = data["generated_text"]
        
        return {
            "text": text,
            "model": self.model,
            "usage": {}
        }
