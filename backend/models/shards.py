# FILE: backend/models/shards.py
"""
Shard models
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class BuildShardRequest(BaseModel):
    """Request to build shard"""
    book_id: str
    chapter_id: str
    pdf_path: str
    seed: Optional[int] = 42
    force_rebuild: bool = False


class ShardInfo(BaseModel):
    """Shard information"""
    book_id: str
    chapter_id: str
    index_type: str
    num_chunks: int
    embedding_dim: int
    created_at: str
    params: Dict[str, Any]
