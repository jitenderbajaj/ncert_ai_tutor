# FILE: backend/routes/ingest_pdf.py
"""
PDF ingestion endpoint
"""
import asyncio
import json
import logging
import os
import threading
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from backend.services.ingestion_dual import ingest_chapter_dual
from backend.config import get_settings

# Job registry (in-memory)
from backend.services.ingest_job_registry import job_registry


logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    book_id: str = Form(...),
    chapter_id: str = Form(...),
    seed: Optional[int] = Form(42)
):
    """
    Ingest PDF with dual indices and image extraction (synchronous).
    """
    logger.info(f"Ingesting PDF: book_id={book_id}, chapter_id={chapter_id}")

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run ingestion
        result = ingest_chapter_dual(
            pdf_path=tmp_path,
            book_id=book_id,
            chapter_id=chapter_id,
            seed=seed
        )

        # Extract manifests
        detail_manifest = result.get("detail_manifest", {})
        summary_manifest = result.get("summary_manifest", {})
        images = result.get("images", [])

        # Calculate counts - use stats.num_chunks
        detail_count = 0
        if "stats" in detail_manifest and "num_chunks" in detail_manifest["stats"]:
            detail_count = detail_manifest["stats"]["num_chunks"]

        summary_count = 0
        if "stats" in summary_manifest and "num_chunks" in summary_manifest["stats"]:
            summary_count = summary_manifest["stats"]["num_chunks"]

        # Get summary text
        summary_text = result.get("summary_text", "")

        return {
            "status": "success",
            "book_id": book_id,
            "chapter_id": chapter_id,
            "seed": seed,

            # Explicit counts (for Streamlit metrics)
            "detail_count": detail_count,
            "summary_count": summary_count,
            "image_count": len(images),
            "summary_text": summary_text,

            # Full manifests
            "detail_manifest": detail_manifest,
            "summary_manifest": summary_manifest,
            "images": images
        }

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/pdf/async")
async def ingest_pdf_async(
    file: UploadFile = File(...),
    book_id: str = Form(...),
    chapter_id: str = Form(...),
    seed: Optional[int] = Form(42)
):
    """
    Ingest PDF asynchronously (job-based).
    Returns immediately with job_id; progress is published to the job registry via emit(...).
    """
    logger.info(f"Async ingest requested: book_id={book_id}, chapter_id={chapter_id}")

    job = job_registry.create_job()
    job_registry.set_status(job.job_id, "queued")
    job_registry.publish(job.job_id, {
        "type": "ingest_job_created",
        "job_id": job.job_id,
        "book_id": book_id,
        "chapter_id": chapter_id,
        "seed": seed,
    })

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    def _run() -> None:
        job_registry.set_status(job.job_id, "running")

        def emit(evt):
            evt = dict(evt or {})
            evt["job_id"] = job.job_id
            evt.setdefault("book_id", book_id)
            evt.setdefault("chapter_id", chapter_id)
            job_registry.publish(job.job_id, evt)

        try:
            result = ingest_chapter_dual(
                pdf_path=tmp_path,
                book_id=book_id,
                chapter_id=chapter_id,
                seed=seed,
                emit=emit,  # requires ingest_chapter_dual to accept emit
            )
            job_registry.complete(job.job_id, result)
        except Exception as e:
            job_registry.fail(job.job_id, str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "accepted",
        "job_id": job.job_id,
        "book_id": book_id,
        "chapter_id": chapter_id,
        "seed": seed,
    }


@router.get("/jobs/{job_id}")
async def get_ingest_job(job_id: str):
    """
    Get job status/result/error.
    """
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result": job.result,
        "error": job.error,
    }


@router.get("/jobs/{job_id}/events")
async def stream_ingest_events(job_id: str):
    """
    Server-Sent Events (SSE) stream for ingestion job progress.
    """
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    async def event_generator():
        # Immediately tell the client current status (helps late subscribers)
        init_evt = {
            "type": "ingest_status",
            "job_id": job.job_id,
            "status": job.status,
        }
        yield f"data: {json.dumps(init_evt, ensure_ascii=False)}\n\n"

        while True:
            # job.events.get() is blocking; run it off the event loop
            evt = await asyncio.to_thread(job.events.get)
            if evt is None:
                break
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )



