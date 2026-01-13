# FILE: backend/routes/textbooks.py
"""
Textbook endpoints
"""
import logging
from typing import List, Optional
from fastapi import APIRouter
from pathlib import Path

from backend.config import get_settings
from backend.services.shard_store import ShardStore

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

shard_store = ShardStore()


@router.get("/list")
async def list_textbooks():
    """List available textbooks"""
    books = shard_store.list_books()
    
    return {
        "books": books
    }


@router.get("/{book_id}/chapters")
async def list_chapters(book_id: str):
    """List chapters for a book"""
    chapters = shard_store.list_chapters(book_id)
    
    return {
        "book_id": book_id,
        "chapters": chapters
    }


@router.get("/{book_id}/{chapter_id}/info")
async def chapter_info(book_id: str, chapter_id: str):
    """Get chapter information"""
    info = shard_store.get_chapter_info(book_id, chapter_id)
    
    return {
        "book_id": book_id,
        "chapter_id": chapter_id,
        "info": info
    }
