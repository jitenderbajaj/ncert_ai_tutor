# FILE: backend/router/router.py
"""
Router logic for provider selection
"""
import logging
from typing import Dict, Any, List, Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def select_provider(
    available_providers: List[str],
    policy: str,
    correlation_id: Optional[str] = None
) -> str:
    """
    Select provider based on policy
    
    Args:
        available_providers: List of available provider names
        policy: offline_first, online_first, or round_robin
        correlation_id: Correlation ID for logging
    
    Returns:
        Selected provider name
    """
    logger.debug(f"[{correlation_id}] Selecting provider with policy={policy}")
    
    offline_providers = ["lmstudio", "ollama"]
    online_providers = ["openai", "openrouter", "huggingface", "grok", "gemini"]
    
    if policy == "offline_first":
        # Prefer offline, then online
        for provider in offline_providers:
            if provider in available_providers:
                return provider
        for provider in online_providers:
            if provider in available_providers:
                return provider
    
    elif policy == "online_first":
        # Prefer online, then offline
        for provider in online_providers:
            if provider in available_providers:
                return provider
        for provider in offline_providers:
            if provider in available_providers:
                return provider
    
    elif policy == "round_robin":
        # Stub: would implement round-robin state
        if available_providers:
            return available_providers[0]
    
    raise RuntimeError("No available providers")
