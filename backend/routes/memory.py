# FILE: backend/routes/memory.py
"""
Memory endpoints
"""
import logging
from typing import Optional, Any
from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.memory_store import MemoryStore
from backend.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

memory_store = MemoryStore()


class MemoryPutRequest(BaseModel):
    """Memory put request"""
    user_id: str
    chapter_id: Optional[str] = None
    key: str
    value: Any
    retention_ttl_days: int = 90


class MemoryGetRequest(BaseModel):
    """Memory get request"""
    user_id: str
    chapter_id: Optional[str] = None
    key: str


@router.post("/put")
async def put_memory(request: MemoryPutRequest):
    """Store memory entry"""
    memory_store.put(
        user_id=request.user_id,
        chapter_id=request.chapter_id,
        key=request.key,
        value=request.value,
        retention_ttl_days=request.retention_ttl_days
    )
    
    return {
        "status": "success",
        "user_id": request.user_id,
        "key": request.key
    }


@router.get("/get")
async def get_memory(user_id: str, key: str, chapter_id: Optional[str] = None):
    """Retrieve memory entry"""
    value = memory_store.get(
        user_id=user_id,
        chapter_id=chapter_id,
        key=key
    )
    
    if value is None:
        return {
            "status": "not_found",
            "user_id": user_id,
            "key": key
        }
    
    return {
        "status": "success",
        "user_id": user_id,
        "key": key,
        "value": value
    }
