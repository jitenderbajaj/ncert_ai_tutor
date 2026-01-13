# FILE: backend/services/shard_store.py
"""
Shard store for managing shards
"""
import logging
from typing import List, Dict, Any
from pathlib import Path
import json

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ShardStore:
    """Store for managing FAISS shards"""
    
    def __init__(self):
        self.shards_dir = Path(settings.shards_dir)
    
    def list_books(self) -> List[str]:
        """List available books"""
        books = set()
        for shard_dir in self.shards_dir.iterdir():
            if shard_dir.is_dir():
                parts = shard_dir.name.split('_')
                if len(parts) >= 2:
                    books.add(parts[0])
        
        return sorted(list(books))
    
    def list_chapters(self, book_id: str) -> List[str]:
        """List chapters for a book"""
        chapters = set()
        for shard_dir in self.shards_dir.iterdir():
            if shard_dir.is_dir() and shard_dir.name.startswith(f"{book_id}_"):
                parts = shard_dir.name.split('_', 1)
                if len(parts) >= 2:
                    chapters.add(parts[1])
        
        return sorted(list(chapters))
    
    def get_chapter_info(self, book_id: str, chapter_id: str) -> Dict[str, Any]:
        """Get chapter information"""
        shard_dir = self.shards_dir / f"{book_id}_{chapter_id}"
        
        if not shard_dir.exists():
            return {"error": "Chapter not found"}
        
        # Load manifests
        detail_manifest_path = shard_dir / "detail_manifest.json"
        summary_manifest_path = shard_dir / "summary_manifest.json"
        
        info = {
            "book_id": book_id,
            "chapter_id": chapter_id,
            "indices": {}
        }
        
        if detail_manifest_path.exists():
            with open(detail_manifest_path, 'r') as f:
                info["indices"]["detail"] = json.load(f)
        
        if summary_manifest_path.exists():
            with open(summary_manifest_path, 'r') as f:
                info["indices"]["summary"] = json.load(f)
        
        return info
