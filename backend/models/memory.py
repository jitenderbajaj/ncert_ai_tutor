# FILE: backend/models/memory.py
"""
Memory models
"""
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class MemoryPut(BaseModel):
    """Store memory entry"""
    user_id: str
    chapter_id: Optional[str] = None
    key: str
    value: Any
    retention_ttl_days: int = Field(default=90, ge=1)


class MemoryGet(BaseModel):
    """Retrieve memory entry"""
    user_id: str
    chapter_id: Optional[str] = None
    key: str
