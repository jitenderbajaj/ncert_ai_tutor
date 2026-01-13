# FILE: backend/providers/openai.py
"""
OpenAI provider adapter
"""
import logging
from typing import Dict, Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI provider"""
    
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"OpenAI provider: model={model}")
    
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate text using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        text = response.choices[0].message.content
        
        return {
            "text": text,
            "model": self.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
