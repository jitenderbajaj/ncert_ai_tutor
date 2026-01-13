# FILE: backend/routes/attempts.py
"""
Attempts endpoints
"""
import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.attempt_store import AttemptStore
from backend.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

attempt_store = AttemptStore()


class AttemptSubmitRequest(BaseModel):
    """Submit attempt request"""
    attempt_id: str
    user_id: str
    question_id: str
    book_id: str
    chapter_id: str
    response: str
    correctness: Optional[float] = None
    bloom_level: Optional[str] = None
    hots_level: Optional[str] = None
    evaluation: Optional[dict] = None


@router.post("/submit")
async def submit_attempt(request: AttemptSubmitRequest):
    """Submit student attempt (idempotent)"""
    logger.info(f"Submit attempt: {request.attempt_id}")
    
    attempt_store.submit(
        attempt_id=request.attempt_id,
        user_id=request.user_id,
        question_id=request.question_id,
        book_id=request.book_id,
        chapter_id=request.chapter_id,
        response=request.response,
        correctness=request.correctness,
        bloom_level=request.bloom_level,
        hots_level=request.hots_level,
        evaluation=request.evaluation
    )
    
    return {
        "status": "success",
        "attempt_id": request.attempt_id
    }


@router.get("/export")
async def export_attempts(
    book_id: Optional[str] = None,
    chapter_id: Optional[str] = None,
    user_id: Optional[str] = None,
    format: str = "csv"
):
    """Export attempts"""
    logger.info(f"Export attempts: format={format}")
    
    attempts = attempt_store.export(
        book_id=book_id,
        chapter_id=chapter_id,
        user_id=user_id,
        format=format
    )
    
    return {
        "status": "success",
        "format": format,
        "count": len(attempts),
        "data": attempts
    }
