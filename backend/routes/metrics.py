# FILE: backend/routes/metrics.py
"""
Global metrics API over rotated JSONL telemetry logs.

Endpoints:
- GET /metrics/summary
- GET /metrics/agents
- GET /metrics/runs

Telemetry files:
  {LOGS_DIR}/telemetry/events-YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, Query
from backend.config import get_settings

logger = logging.getLogger(__name__)
# router = APIRouter(prefix="/metrics", tags=["metrics"])
router = APIRouter(tags=["metrics"])
settings = get_settings()


# ----------------------------
# Helpers
# ----------------------------

def _parse_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if not isinstance(ts, str):
        return None

    s = ts.strip()
    # support "...Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    vals = sorted(values)
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    return float(vals[f] + (vals[c] - vals[f]) * (k - f))


def _telemetry_dir() -> Path:
    logs_dir = Path(getattr(settings, "LOGS_DIR", "./logs"))
    return logs_dir / "telemetry"


def _iter_candidate_files(start_utc: datetime, end_utc: datetime) -> List[Path]:
    """
    Select rotated files by date range.
    Files are named events-YYYY-MM-DD.jsonl.
    We include all dates that intersect [start_utc, end_utc].
    """
    td = _telemetry_dir()
    if not td.exists():
        return []

    # Iterate UTC dates; note: rotation might be based on IST, but files are still daily.
    # Including all UTC dates in range is a safe superset (may read 1 extra file).
    start_date = start_utc.date()
    end_date = end_utc.date()
    files: List[Path] = []
    d = start_date
    while d <= end_date:
        p = td / f"events-{d.isoformat()}.jsonl"
        if p.exists():
            files.append(p)
        d = d + timedelta(days=1)

    # Also include any files not in UTC naming (if you rotate by IST date, file names are IST dates).
    # As a safe fallback, include all files that exist and filter by ts window anyway.
    if not files:
        files = sorted(td.glob("events-*.jsonl"))

    return files


def _iter_events(
    since_hours: int,
    bookid: Optional[str] = None,
    chapterid: Optional[str] = None,
    userid: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=max(1, since_hours))
    end = now

    files = _iter_candidate_files(start, end)
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts = _parse_ts(ev.get("ts") or ev.get("timestamp"))
                    if not ts:
                        continue
                    if ts < start or ts > end:
                        continue

                    if bookid and ev.get("bookid") != bookid:
                        continue
                    if chapterid and ev.get("chapterid") != chapterid:
                        continue
                    if userid and ev.get("userid") != userid:
                        continue

                    yield ev
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning("metrics: failed reading %s: %s", str(path), e)
            continue


def _normalize_event_name(ev: Dict[str, Any]) -> str:
    # telemetry.py uses "event"; legacy telemetry used "type"
    return str(ev.get("event") or ev.get("type") or "").strip()


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ----------------------------
# API: /metrics/summary
# ----------------------------

@router.get("/summary")
def metrics_summary(
    since_hours: int = Query(24, ge=1, le=24 * 30),
    bookid: Optional[str] = None,
    chapterid: Optional[str] = None,
    userid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level summary for dashboard KPIs.
    """
    total_events = 0
    counts: Dict[str, int] = {}
    run_durations: List[float] = []
    verdict_counts: Dict[str, int] = {}
    strategy_counts: Dict[str, int] = {}
    reflection_true = 0
    reflection_total = 0

    retrieve_results: List[int] = []
    zero_retrieval = 0
    retrieve_events = 0

    for ev in _iter_events(since_hours, bookid=bookid, chapterid=chapterid, userid=userid):
        total_events += 1
        name = _normalize_event_name(ev)
        counts[name] = counts.get(name, 0) + 1

        if name == "agent_graph_complete":
            dur = _safe_int(ev.get("durationms"), 0)
            if dur > 0:
                run_durations.append(float(dur))

            verdict = str(ev.get("verdict") or "unknown").lower()
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

            strat = ev.get("strategy")
            if strat:
                s = str(strat).lower()
                strategy_counts[s] = strategy_counts.get(s, 0) + 1

            # if you store needsrefinement at graph_complete
            if "needsrefinement" in ev:
                reflection_total += 1
                if bool(ev.get("needsrefinement")):
                    reflection_true += 1

        if name == "agent_step":
            # detect retrieve step results counts if present
            agent = str(ev.get("agent") or "").lower()
            if agent in {"retrieve", "retrieverefined"}:
                retrieve_events += 1
                rc = ev.get("resultscount")
                if rc is not None:
                    rci = _safe_int(rc, 0)
                    retrieve_results.append(rci)
                    if rci == 0:
                        zero_retrieval += 1

            # planner strategy if present in fields
            if agent == "planner":
                strat = ev.get("strategy") or (ev.get("output") or {}).get("strategy")
                if strat:
                    s = str(strat).lower()
                    strategy_counts[s] = strategy_counts.get(s, 0) + 1

            # reflect needsrefinement if present in fields
            if agent == "reflect" and "needsrefinement" in ev:
                reflection_total += 1
                if bool(ev.get("needsrefinement")):
                    reflection_true += 1

    summary = {
        "since_hours": since_hours,
        "filters": {"bookid": bookid, "chapterid": chapterid, "userid": userid},
        "total_events": total_events,
        "event_counts": counts,
        "runs": {
            "count": counts.get("agent_graph_complete", 0),
            "avg_duration_ms": (sum(run_durations) / len(run_durations)) if run_durations else 0.0,
            "p50_duration_ms": _percentile(run_durations, 50),
            "p95_duration_ms": _percentile(run_durations, 95),
            "max_duration_ms": max(run_durations) if run_durations else 0.0,
        },
        "routing": {
            "strategy_counts": strategy_counts,
        },
        "retrieval": {
            "retrieve_events_with_counts": len(retrieve_results),
            "avg_resultscount": (sum(retrieve_results) / len(retrieve_results)) if retrieve_results else 0.0,
            "zero_result_rate": (zero_retrieval / len(retrieve_results)) if retrieve_results else 0.0,
        },
        "reflection": {
            "observations": reflection_total,
            "rate": (reflection_true / reflection_total) if reflection_total else 0.0,
        },
        "governance": {
            "verdict_counts": verdict_counts,
        },
    }
    return summary


# ----------------------------
# API: /metrics/agents
# ----------------------------

@router.get("/agents")
def metrics_agents(
    since_hours: int = Query(24, ge=1, le=24 * 30),
    bookid: Optional[str] = None,
    chapterid: Optional[str] = None,
    userid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Per-agent latency aggregates from agent_step events.
    """
    per_agent: Dict[str, List[float]] = {}
    per_agent_counts: Dict[str, int] = {}

    for ev in _iter_events(since_hours, bookid=bookid, chapterid=chapterid, userid=userid):
        if _normalize_event_name(ev) != "agent_step":
            continue
        agent = str(ev.get("agent") or "unknown").lower()
        dur = _safe_int(ev.get("durationms"), 0)
        if dur < 0:
            continue
        per_agent.setdefault(agent, []).append(float(dur))
        per_agent_counts[agent] = per_agent_counts.get(agent, 0) + 1

    rows: List[Dict[str, Any]] = []
    for agent, durs in per_agent.items():
        rows.append(
            {
                "agent": agent,
                "count": per_agent_counts.get(agent, len(durs)),
                "avg_duration_ms": sum(durs) / len(durs) if durs else 0.0,
                "p50_duration_ms": _percentile(durs, 50),
                "p95_duration_ms": _percentile(durs, 95),
                "max_duration_ms": max(durs) if durs else 0.0,
            }
        )

    rows.sort(key=lambda r: (r["p95_duration_ms"], r["avg_duration_ms"]), reverse=True)

    return {
        "since_hours": since_hours,
        "filters": {"bookid": bookid, "chapterid": chapterid, "userid": userid},
        "agents": rows,
    }


# ----------------------------
# API: /metrics/runs
# ----------------------------

@router.get("/runs")
def metrics_runs(
    limit: int = Query(50, ge=1, le=500),
    since_hours: int = Query(24, ge=1, le=24 * 30),
    bookid: Optional[str] = None,
    chapterid: Optional[str] = None,
    userid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Recent run list based on agent_graph_complete events.
    """
    runs: List[Dict[str, Any]] = []

    for ev in _iter_events(since_hours, bookid=bookid, chapterid=chapterid, userid=userid):
        if _normalize_event_name(ev) != "agent_graph_complete":
            continue

        runs.append(
            {
                "ts": ev.get("ts") or ev.get("timestamp"),
                "correlationid": ev.get("correlationid"),
                "userid": ev.get("userid"),
                "bookid": ev.get("bookid"),
                "chapterid": ev.get("chapterid"),
                "durationms": _safe_int(ev.get("durationms"), 0),
                "verdict": ev.get("verdict", "unknown"),
                "strategy": ev.get("strategy"),
                "indexhint": ev.get("indexhint"),
                "needsrefinement": ev.get("needsrefinement"),
                "doccount": ev.get("doccount"),
                "resultscount": ev.get("resultscount"),
            }
        )

    # sort newest first
    def _sort_key(r: Dict[str, Any]) -> Tuple[int, str]:
        ts = _parse_ts(r.get("ts"))
        epoch = int(ts.timestamp()) if ts else 0
        return (epoch, str(r.get("correlationid") or ""))

    runs.sort(key=_sort_key, reverse=True)
    runs = runs[:limit]

    return {
        "since_hours": since_hours,
        "limit": limit,
        "filters": {"bookid": bookid, "chapterid": chapterid, "userid": userid},
        "runs": runs,
    }
