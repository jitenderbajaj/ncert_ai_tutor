# FILE: backend/models/assessments.py
"""
Assessment models
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AssessmentQuestion(BaseModel):
    """Assessment question schema"""
    question_id: str
    book_id: str
    chapter_id: str
    question_text: str
    question_type: str = Field(..., description="mcq, short_answer, long_answer")
    difficulty: str = Field(..., description="easy, medium, hard")
    bloom_level: str = Field(..., description="remember, understand, apply, analyze, evaluate, create")
    hots_level: str = Field(..., description="easy, medium, hard")
    rubric_key: Optional[str] = None
    correct_answer: Optional[str] = None
    options: Optional[List[str]] = None
    hints: Optional[List[str]] = None
    explanation: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    bilingual_notes: Optional[Dict[str, str]] = None


class GenerateAssessmentRequest(BaseModel):
    """Request to generate assessment questions"""
    book_id: str
    chapter_id: str
    count: int = Field(default=5, ge=1, le=20)
    difficulty: Optional[str] = None
    question_type: Optional[str] = None
