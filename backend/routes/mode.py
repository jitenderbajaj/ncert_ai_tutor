# FILE: backend/routes/mode.py
"""
Mode status endpoint
"""
import logging
from fastapi import APIRouter

from backend.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("")
async def get_mode():
    """Get current mode status"""
    return {
        "mode": settings.llm_mode,
        "router_policy": settings.router_policy,
        "router_fallback": settings.router_fallback
    }
