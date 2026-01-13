# FILE: backend/services/manifest_store.py
"""
Manifest store
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ManifestStore:
    """Store for shard manifests"""
    
    def __init__(self):
        self.shards_dir = Path(settings.shards_dir)
    
    def get_manifest(
        self,
        book_id: str,
        chapter_id: str,
        index_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get manifest for shard"""
        shard_dir = self.shards_dir / f"{book_id}_{chapter_id}"
        manifest_path = shard_dir / f"{index_type}_manifest.json"
        
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def put_manifest(
        self,
        book_id: str,
        chapter_id: str,
        index_type: str,
        manifest: Dict[str, Any]
    ):
        """Store manifest"""
        shard_dir = self.shards_dir / f"{book_id}_{chapter_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = shard_dir / f"{index_type}_manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
