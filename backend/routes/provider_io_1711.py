# FILE: backend/routes/provider_io.py

"""
Provider I/O endpoints for debugging and transparency
"""
import logging
from typing import Optional
from fastapi import APIRouter

from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/recent")
async def get_recent_io(limit: int = 20):
    """Get recent provider I/O entries"""
    registry = get_provider_registry()
    entries = registry.get_recent_io(limit=limit)
    
    return {
        "status": "success",
        "count": len(entries),
        "entries": entries
    }


@router.post("/clear")
async def clear_io_log():
    """Clear in-memory I/O log"""
    registry = get_provider_registry()
    registry.clear_io_log()
    
    return {
        "status": "success",
        "message": "I/O log cleared"
    }
