# FILE: backend/services/retrieve_cache.py
"""
Chapter-scoped retrieve cache with deterministic keys
"""
import logging
from typing import Dict, Any, List, Optional
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RetrieveCache:
    """Retrieve cache with TTL and deterministic keys"""
    
    def __init__(self):
        self.cache_dir = Path(settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = settings.cache_ttl_hours
    
    def _get_key(self, query: str, book_id: str, chapter_id: str, index_type: str) -> str:
        """Generate deterministic cache key"""
        key_str = f"{query}|{book_id}|{chapter_id}|{index_type}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        book_id: str,
        chapter_id: str,
        index_type: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached results"""
        key = self._get_key(query, book_id, chapter_id, index_type)
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check TTL
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=self.ttl_hours):
            logger.debug(f"Cache expired: {key}")
            cache_file.unlink()
            return None
        
        # Load cache
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        logger.debug(f"Cache hit: {key}")
        return data["results"]
    
    def put(
        self,
        query: str,
        book_id: str,
        chapter_id: str,
        index_type: str,
        results: List[Dict[str, Any]]
    ):
        """Store results in cache"""
        key = self._get_key(query, book_id, chapter_id, index_type)
        cache_file = self.cache_dir / f"{key}.json"
        
        data = {
            "query": query,
            "book_id": book_id,
            "chapter_id": chapter_id,
            "index_type": index_type,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        logger.debug(f"Cache stored: {key}")
    
    def warm(self, book_id: str, chapter_id: str):
        """Warm cache for common queries"""
        # Stub: would pre-cache common queries
        logger.info(f"Warming cache for {book_id}/{chapter_id}")
    
    def get_stats(self, book_id: str, chapter_id: str) -> Dict[str, Any]:
        """Get cache statistics"""
        # Stub: would count cache files for book/chapter
        cache_files = list(self.cache_dir.glob("*.json"))
        
        return {
            "total_entries": len(cache_files),
            "ttl_hours": self.ttl_hours
        }
    
    def clear(self, book_id: Optional[str] = None, chapter_id: Optional[str] = None):
        """Clear cache"""
        if book_id is None:
            # Clear all
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            logger.info("Cleared all cache")
        else:
            # Clear for specific book/chapter
            # Stub: would filter by book_id/chapter_id
            logger.info(f"Cleared cache for {book_id}/{chapter_id}")
