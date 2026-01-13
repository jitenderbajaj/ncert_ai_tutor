# FILE: backend/models/retrieve.py
"""
Retrieve models
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Retrieve passages request"""
    query: str
    book_id: str
    chapter_id: str
    index_type: str = Field(default="detail", description="detail or summary")
    top_k: int = Field(default=5, ge=1, le=20)
