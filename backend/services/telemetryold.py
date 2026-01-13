# FILE: backend/services/telemetry.py
"""
Telemetry and metrics collection
"""
import logging
from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


# Global telemetry store (in-memory for simplicity)
_telemetry_events: List[Dict[str, Any]] = []
_telemetry_counters: Dict[str, int] = defaultdict(int)


def init_telemetry():
    """Initialize telemetry"""
    logger.info("Telemetry initialized")


def record_event(event_type: str, data: Dict[str, Any]):
    """Record telemetry event"""
    event = {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }
    _telemetry_events.append(event)
    _telemetry_counters[event_type] += 1


def get_telemetry_summary() -> Dict[str, Any]:
    """Get telemetry summary"""
    return {
        "total_events": len(_telemetry_events),
        "counters": dict(_telemetry_counters),
        "recent_events": _telemetry_events[-10:]
    }
