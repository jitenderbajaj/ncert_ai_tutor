# FILE: backend/services/diagnostics.py
"""
Diagnostic utilities
"""
import logging
from typing import Dict, Any
from pathlib import Path

from backend.config import get_settings
from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)
settings = get_settings()


def run_diagnostics() -> Dict[str, Any]:
    """Run system diagnostics"""
    diagnostics = {
        "config": check_config(),
        "providers": check_providers(),
        "data_dirs": check_data_dirs(),
        "shards": check_shards()
    }
    
    return diagnostics


def check_config() -> Dict[str, Any]:
    """Check configuration"""
    return {
        "llm_mode": settings.llm_mode,
        "router_policy": settings.router_policy,
        "data_dir": str(settings.data_dir),
        "status": "ok"
    }


def check_providers() -> Dict[str, Any]:
    """Check provider availability"""
    registry = get_provider_registry()
    
    return {
        "available": list(registry.providers.keys()),
        "count": len(registry.providers),
        "status": "ok" if registry.providers else "error"
    }


def check_data_dirs() -> Dict[str, Any]:
    """Check data directories"""
    dirs_exist = {
        "shards": Path(settings.shards_dir).exists(),
        "summaries": Path(settings.summaries_dir).exists(),
        "images": Path(settings.images_dir).exists(),
        "memory": Path(settings.memory_dir).exists(),
        "cache": Path(settings.cache_dir).exists(),
        "attempts": Path(settings.attempts_dir).exists()
    }
    
    return {
        "dirs": dirs_exist,
        "status": "ok" if all(dirs_exist.values()) else "warning"
    }


def check_shards() -> Dict[str, Any]:
    """Check shard availability"""
    shards_dir = Path(settings.shards_dir)
    
    if not shards_dir.exists():
        return {"count": 0, "status": "warning"}
    
    shard_dirs = [d for d in shards_dir.iterdir() if d.is_dir()]
    
    return {
        "count": len(shard_dirs),
        "status": "ok" if shard_dirs else "warning"
    }
