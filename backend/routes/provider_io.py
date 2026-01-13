# FILE: backend/routes/provider_io.py (UPDATED WITH ROBUST ERROR HANDLING)
"""
Provider I/O endpoints for debugging and transparency
Fixed: Handles invalid entries gracefully
"""
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Query
from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/recent")
async def get_recent_io(limit: int = Query(default=20, ge=1, le=100)):
    """
    Get recent provider I/O entries with robust error handling
    
    Filters out invalid entries and ensures all returned data is properly formatted
    """
    try:
        registry = get_provider_registry()
        raw_entries = registry.get_recent_io(limit=limit)
        
        # Validate and sanitize entries
        valid_entries = []
        
        for entry in raw_entries:
            try:
                # Skip non-dict entries
                if not isinstance(entry, dict):
                    logger.warning(f"[PROVIDER_IO] Skipping non-dict entry: {type(entry)}")
                    continue
                
                # Create sanitized entry with safe defaults
                sanitized_entry = {
                    "timestamp": entry.get("timestamp", "unknown"),
                    "correlation_id": entry.get("correlation_id", ""),
                    "provider": entry.get("provider", "unknown"),
                    "model": entry.get("model", "unknown"),
                    # "prompt": entry.get("prompt_sanitized", entry.get("prompt", "")),
                    "prompt": entry.get("prompt", entry.get("prompt_sanitized", "")),
                    "output": entry.get("output", ""),
                    "duration_ms": entry.get("duration_ms", 0),
                    "error": entry.get("error", None)
                }
                
                valid_entries.append(sanitized_entry)
                
            except Exception as e:
                logger.error(f"[PROVIDER_IO] Error processing entry: {e}")
                continue
        
        return {
            "status": "success",
            "count": len(valid_entries),
            "entries": valid_entries
        }
    
    except Exception as e:
        logger.error(f"[PROVIDER_IO] Error retrieving I/O entries: {e}")
        return {
            "status": "error",
            "count": 0,
            "entries": [],
            "error": str(e)
        }

@router.post("/clear")
async def clear_io_log():
    """
    Clear in-memory I/O log
    """
    try:
        registry = get_provider_registry()
        registry.clear_io_log()
        
        return {
            "status": "success",
            "message": "I/O log cleared"
        }
    
    except Exception as e:
        logger.error(f"[PROVIDER_IO] Error clearing log: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@router.get("/stats")
async def get_io_stats():
    """
    Get statistics about provider I/O usage
    """
    try:
        registry = get_provider_registry()
        entries = registry.get_recent_io(limit=1000)
        
        # Count by provider
        provider_counts = {}
        error_count = 0
        total_duration = 0
        
        for entry in entries:
            if isinstance(entry, dict):
                provider = entry.get("provider", "unknown")
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
                
                if entry.get("error"):
                    error_count += 1
                
                total_duration += entry.get("duration_ms", 0)
        
        return {
            "status": "success",
            "total_calls": len(entries),
            "error_count": error_count,
            "success_rate": (len(entries) - error_count) / max(1, len(entries)),
            "avg_duration_ms": total_duration / max(1, len(entries)),
            "provider_counts": provider_counts
        }
    
    except Exception as e:
        logger.error(f"[PROVIDER_IO] Error getting stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
