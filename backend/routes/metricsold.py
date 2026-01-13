# FILE: backend/routes/metrics.py
"""
Metrics endpoint
"""
import logging
from fastapi import APIRouter

from backend.services.telemetry import get_telemetry_summary

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def get_metrics():
    """Get telemetry metrics"""
    summary = get_telemetry_summary()
    
    return {
        "status": "success",
        "metrics": summary
    }
