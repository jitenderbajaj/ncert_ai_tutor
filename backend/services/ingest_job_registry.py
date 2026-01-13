# backend/services/ingest_job_registry.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4
import time
import queue
import threading


@dataclass
class IngestJob:
    job_id: str
    status: str = "queued"  # queued | running | success | error
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Thread-safe event channel for SSE consumers
    events: "queue.Queue[Optional[Dict[str, Any]]]" = field(default_factory=queue.Queue)


class IngestJobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, IngestJob] = {}

    def create_job(self) -> IngestJob:
        job = IngestJob(job_id=str(uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.status = status
        job.updated_at = time.time()

    def publish(self, job_id: str, event: Dict[str, Any]) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.updated_at = time.time()
        job.events.put(event)

    def complete(self, job_id: str, result: Dict[str, Any]) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.status = "success"
        job.result = result
        job.updated_at = time.time()
        job.events.put({"type": "ingestjob_complete", "job_id": job_id, "pct": 100, "status": "success", "result": result})
        job.events.put(None)  # sentinel: close SSE stream

    def fail(self, job_id: str, error: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.status = "error"
        job.error = error
        job.updated_at = time.time()
        job.events.put({"type": "ingest_error", "job_id": job_id, "error": error})
        job.events.put(None)  # sentinel: close SSE stream


# Global singleton (simple in-memory approach)
job_registry = IngestJobRegistry()
