# FILE: backend/providers/offline.py
"""
Base class for offline providers
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class OfflineProvider:
    """Base class for offline LLM providers"""
    
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        raise NotImplementedError
