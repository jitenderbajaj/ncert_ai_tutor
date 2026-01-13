# FILE: backend/services/startup_verify.py
"""
Startup verification
"""
import logging
from typing import Dict, Any

from backend.config import get_settings
from backend.providers.registry import get_provider_registry
from backend.services.diagnostics import run_diagnostics

logger = logging.getLogger(__name__)
settings = get_settings()


async def verify_startup() -> Dict[str, Any]:
    """Verify system startup requirements"""
    logger.info("Running startup verification")
    
    # Check LLM providers
    registry = get_provider_registry()
    llm_active = len(registry.providers) > 0
    
    if not llm_active:
        logger.error("No LLM providers available - LLM-mandatory requirement not met")
        return {
            "llm_active": False,
            "mode": settings.llm_mode,
            "error": "No LLM providers available"
        }
    
    # Run diagnostics
    diagnostics = run_diagnostics()
    
    logger.info(f"Startup verification passed: {len(registry.providers)} providers available")
    
    return {
        "llm_active": True,
        "mode": settings.llm_mode,
        "providers": list(registry.providers.keys()),
        "diagnostics": diagnostics
    }
