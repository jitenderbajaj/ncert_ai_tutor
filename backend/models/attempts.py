# FILE: backend/models/attempts.py
"""
Attempt models
"""
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AttemptSubmit(BaseModel):
    """Submit student attempt"""
    attempt_id: str
    user_id: str
    question_id: str
    book_id: str
    chapter_id: str
    response: str
    correctness: Optional[float] = Field(None, ge=0.0, le=1.0)
    bloom_level: Optional[str] = None
    hots_level: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class AttemptExportRequest(BaseModel):
    """Request to export attempts"""
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None
    user_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    format: str = Field(default="csv", description="csv or json")
