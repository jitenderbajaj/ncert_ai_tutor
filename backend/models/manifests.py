# FILE: backend/models/manifests.py
"""
Manifest models
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class ShardManifest(BaseModel):
    """Shard manifest schema"""
    book_id: str
    chapter_id: str
    index_type: str  # detail or summary
    shard_version: str
    created_at: str
    params: Dict[str, Any]
    checksums: Dict[str, str]
    stats: Dict[str, Any]


class ImageManifestEntry(BaseModel):
    """Image manifest entry in images.jsonl"""
    image_id: str
    book_id: str
    chapter_id: str
    page: int
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    path: str
