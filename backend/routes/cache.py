# FILE: backend/routes/cache.py
"""
Cache endpoints
"""
import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.retrieve_cache import RetrieveCache
from backend.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

cache = RetrieveCache()


class CacheWarmRequest(BaseModel):
    """Cache warm request"""
    book_id: str
    chapter_id: str


@router.post("/warm")
async def warm_cache(request: CacheWarmRequest):
    """Warm cache for book/chapter"""
    logger.info(f"Warming cache: {request.book_id}/{request.chapter_id}")
    
    # Stub: would pre-load common queries
    cache.warm(request.book_id, request.chapter_id)
    
    return {
        "status": "success",
        "book_id": request.book_id,
        "chapter_id": request.chapter_id
    }


@router.get("/status")
async def cache_status(book_id: str, chapter_id: str):
    """Get cache statistics"""
    stats = cache.get_stats(book_id, chapter_id)
    
    return {
        "book_id": book_id,
        "chapter_id": chapter_id,
        "stats": stats
    }


@router.post("/clear")
async def clear_cache(book_id: Optional[str] = None, chapter_id: Optional[str] = None):
    """Clear cache"""
    cache.clear(book_id, chapter_id)
    
    return {
        "status": "success",
        "cleared": "all" if not book_id else f"{book_id}/{chapter_id}"
    }
