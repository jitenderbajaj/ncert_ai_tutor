# FILE: backend/routes/health.py
"""
Health check endpoint
"""
import logging
from fastapi import APIRouter, Header
from typing import Optional

from backend.config import get_settings
from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("")
async def health_check(x_mode: Optional[str] = Header(None)):
    """
    Health check endpoint
    Returns llm_active=true if LLM providers are available
    """
    registry = get_provider_registry()
    
    llm_active = len(registry.providers) > 0
    
    return {
        "status": "healthy",
        "version": "0.11.0",
        "mode": x_mode or settings.llm_mode,
        "llm_active": llm_active,
        "available_providers": list(registry.providers.keys())
    }
