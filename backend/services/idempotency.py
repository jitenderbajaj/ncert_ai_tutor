# FILE: backend/services/idempotency.py
"""
Idempotency utilities
"""
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IdempotencyStore:
    """Store for idempotency keys"""
    
    def __init__(self, store_dir: str = "./data/idempotency"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = 24
    
    def check(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """Check if request with key already processed"""
        key_file = self.store_dir / f"{idempotency_key}.json"
        
        if not key_file.exists():
            return None
        
        with open(key_file, 'r') as f:
            data = json.load(f)
        
        # Check TTL
        created_at = datetime.fromisoformat(data["created_at"])
        if datetime.utcnow() - created_at > timedelta(hours=self.ttl_hours):
            key_file.unlink()
            return None
        
        return data["response"]
    
    def store(self, idempotency_key: str, response: Dict[str, Any]):
        """Store response for idempotency key"""
        key_file = self.store_dir / f"{idempotency_key}.json"
        
        data = {
            "idempotency_key": idempotency_key,
            "response": response,
            "created_at": datetime.utcnow().isoformat()
        }
        
        with open(key_file, 'w') as f:
            json.dump(data, f)
