# FILE: backend/services/router.py
"""
Router service utilities
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_router_status() -> Dict[str, Any]:
    """Get router status"""
    from backend.config import get_settings
    from backend.providers.registry import get_provider_registry
    
    settings = get_settings()
    registry = get_provider_registry()
    
    return {
        "policy": settings.router_policy,
        "fallback_enabled": settings.router_fallback,
        "timeout": settings.router_timeout,
        "available_providers": list(registry.providers.keys()),
        "circuit_breaker": {
            "threshold": registry.circuit_breaker.threshold,
            "timeout_seconds": registry.circuit_breaker.timeout_seconds,
            "open_circuits": list(registry.circuit_breaker.open_until.keys())
        }
    }
