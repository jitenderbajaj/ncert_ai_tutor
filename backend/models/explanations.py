# FILE: backend/models/explanations.py
"""
Explanation models
"""
from pydantic import BaseModel


class ExplanationRequest(BaseModel):
    """Request for explanation"""
    question_id: str
    book_id: str
    chapter_id: str
    user_response: str
