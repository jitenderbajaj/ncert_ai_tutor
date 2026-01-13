# FILE: backend/services/telemetry.py
"""
Telemetry and metrics collection (global, rotated JSONL)

- Stores summary-only telemetry events to disk (append-only JSONL).
- Rotates daily based on configurable timezone (UTC in prod, IST in dev).
- Keeps a small in-memory tail for quick local debugging.

Expected env/config keys (via backend.config.getsettings()):
- TELEMETRY_ENABLED: bool
- TELEMETRY_ROTATION: str ("daily")
- TELEMETRY_TIMEZONE: str ("UTC" or "Asia/Kolkata" or any IANA tz)
- TELEMETRY_RETENTION_DAYS: int
- LOGS_DIR: str (base logs dir)
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Small in-memory tail (debug convenience, not the source of truth)
_MAX_IN_MEMORY_EVENTS = 200
_recent_events: Deque[Dict[str, Any]] = deque(maxlen=_MAX_IN_MEMORY_EVENTS)
_counters: Dict[str, int] = defaultdict(int)


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool
    rotation: str
    tz_name: str
    retention_days: int
    logs_dir: Path


def _get_config() -> TelemetryConfig:
    logs_dir = Path(getattr(settings, "LOGS_DIR", "./logs"))
    enabled = bool(getattr(settings, "TELEMETRY_ENABLED", True))
    rotation = str(getattr(settings, "TELEMETRY_ROTATION", "daily")).strip().lower()
    tz_name = str(getattr(settings, "TELEMETRY_TIMEZONE", "UTC")).strip()
    retention_days = int(getattr(settings, "TELEMETRY_RETENTION_DAYS", 90))
    return TelemetryConfig(
        enabled=enabled,
        rotation=rotation,
        tz_name=tz_name,
        retention_days=retention_days,
        logs_dir=logs_dir,
    )


def _resolve_tz(tz_name: str):
    """
    Resolve timezone from IANA name.

    Uses zoneinfo when available (Python 3.9+). Falls back to UTC if invalid.
    """
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        return ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid TELEMETRY_TIMEZONE=%s; falling back to UTC", tz_name)
        return timezone.utc


def _telemetry_dir(cfg: TelemetryConfig) -> Path:
    d = cfg.logs_dir / "telemetry"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _event_file_path(cfg: TelemetryConfig, now_utc: datetime) -> Path:
    """
    Rotation is based on local date in configured timezone, while timestamps remain UTC.
    """
    tz = _resolve_tz(cfg.tz_name)
    local_dt = now_utc.astimezone(tz)
    date_str = local_dt.date().isoformat()  # YYYY-MM-DD

    if cfg.rotation != "daily":
        # Default to daily even if misconfigured
        logger.warning("Unsupported TELEMETRY_ROTATION=%s; using daily", cfg.rotation)

    return _telemetry_dir(cfg) / f"events-{date_str}.jsonl"


def _prune_old_files(cfg: TelemetryConfig) -> None:
    """
    Delete rotated telemetry files older than retention_days.
    Safe best-effort cleanup.
    """
    try:
        retention_days = max(cfg.retention_days, 1)
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        telemetry_dir = _telemetry_dir(cfg)

        for p in telemetry_dir.glob("events-*.jsonl"):
            # Parse filename date (events-YYYY-MM-DD.jsonl)
            name = p.name
            try:
                date_part = name.replace("events-", "").replace(".jsonl", "")
                file_date = datetime.fromisoformat(date_part).replace(tzinfo=timezone.utc)
            except Exception:
                continue

            if file_date < cutoff:
                p.unlink(missing_ok=True)
    except Exception as e:
        logger.debug("Telemetry prune skipped: %s", e)


def init_telemetry() -> None:
    """Initialize telemetry (create dirs + retention prune)."""
    cfg = _get_config()
    if not cfg.enabled:
        logger.info("Telemetry disabled")
        return

    _telemetry_dir(cfg)
    _prune_old_files(cfg)
    logger.info(
        "Telemetry initialized (rotation=%s tz=%s retention_days=%s dir=%s)",
        cfg.rotation,
        cfg.tz_name,
        cfg.retention_days,
        str(cfg.logs_dir),
    )


def record_event(event: str, **fields: Any) -> None:
    """
    Record telemetry event (summary-only).

    NOTE: graph_lg.py imports recordevent directly, so keep this function name stable.
    """
    cfg = _get_config()
    if not cfg.enabled:
        return

    now_utc = datetime.now(timezone.utc)

    payload: Dict[str, Any] = {
        "ts": now_utc.isoformat(),
        "event": event,
        # helpful for audits/debugging of rotation choice
        "tz": cfg.tz_name,
        **fields,
    }
    
    # DEBUG: show agent_step events as they are recorded
    if event == "agent_step":
        logger.info("TELEMETRY_EVENT agent_step %s", payload)

    # In-memory tail
    _recent_events.append(payload)
    _counters[event] += 1

    # Persist to rotated JSONL
    path = _event_file_path(cfg, now_utc)
    try:
        line = json.dumps(payload, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        # Never break the app/graph on telemetry failure
        logger.warning("Failed to write telemetry event to %s: %s", str(path), e)


def get_telemetry_summary() -> Dict[str, Any]:
    """
    Lightweight summary (does NOT scan JSONL files).
    Intended for quick sanity checks and dev debugging.
    """
    return {
        "enabled": bool(getattr(settings, "TELEMETRY_ENABLED", True)),
        "rotation": str(getattr(settings, "TELEMETRY_ROTATION", "daily")),
        "timezone": str(getattr(settings, "TELEMETRY_TIMEZONE", "UTC")),
        "retention_days": int(getattr(settings, "TELEMETRY_RETENTION_DAYS", 90)),
        "total_events_in_memory": len(_recent_events),
        "counters_in_memory": dict(_counters),
        "recent_events": list(_recent_events)[-10:],
    }

