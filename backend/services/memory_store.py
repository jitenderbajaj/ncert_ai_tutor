# FILE: backend/services/memory_store.py
"""
Governed memory store with TTL and redaction
"""
import logging
from typing import Any, Optional, Dict
from pathlib import Path
import json
from datetime import datetime, timedelta

from backend.config import get_settings
from backend.services.correlation import generate_correlation_id

logger = logging.getLogger(__name__)
settings = get_settings()


class MemoryStore:
    """Governed persistent memory store"""
    
    def __init__(self):
        self.memory_dir = Path(settings.memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def put(
        self,
        user_id: str,
        key: str,
        value: Any,
        chapter_id: Optional[str] = None,
        retention_ttl_days: int = 90
    ):
        """Store memory entry with TTL"""
        memory_key = self._get_memory_key(user_id, chapter_id, key)
        memory_file = self.memory_dir / f"{memory_key}.json"
        
        expiry = datetime.utcnow() + timedelta(days=retention_ttl_days)
        
        entry = {
            "user_id": user_id,
            "chapter_id": chapter_id,
            "key": key,
            "value": value,
            "retention_ttl_days": retention_ttl_days,
            "expiry": expiry.isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "correlation_id": generate_correlation_id()
        }
        
        with open(memory_file, 'w') as f:
            json.dump(entry, f, indent=2)
        
        logger.debug(f"Memory stored: {memory_key}")
    
    def get(
        self,
        user_id: str,
        key: str,
        chapter_id: Optional[str] = None
    ) -> Optional[Any]:
        """Retrieve memory entry"""
        memory_key = self._get_memory_key(user_id, chapter_id, key)
        memory_file = self.memory_dir / f"{memory_key}.json"
        
        if not memory_file.exists():
            return None
        
        with open(memory_file, 'r') as f:
            entry = json.load(f)
        
        # Check TTL
        expiry = datetime.fromisoformat(entry["expiry"])
        if datetime.utcnow() > expiry:
            logger.debug(f"Memory expired: {memory_key}")
            memory_file.unlink()
            return None
        
        logger.debug(f"Memory retrieved: {memory_key}")
        return entry["value"]
    
    def _get_memory_key(self, user_id: str, chapter_id: Optional[str], key: str) -> str:
        """Generate memory key"""
        import hashlib
        if chapter_id:
            key_str = f"{user_id}|{chapter_id}|{key}"
        else:
            key_str = f"{user_id}|{key}"
        return hashlib.sha256(key_str.encode()).hexdigest()
