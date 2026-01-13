# FILE: backend/app.py
"""
FastAPI application entry point for NCERT AI Tutor
Increment 11: LLM-mandatory, dual indices, governed memory, visuals
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.middleware.body_limit import BodySizeLimitMiddleware
from backend.middleware.rate_limit import RateLimitMiddleware
from backend.routes import (
    health, mode, agent, ingest_pdf, memory, cache,
    attempts, exports, textbooks, multimodal, visuals, metrics,
    provider_io
)
from backend.services.startup_verify import verify_startup
from backend.services.telemetry import init_telemetry

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    logger.info("Starting NCERT AI Tutor backend v0.11.0")
    
    # Startup verification
    verify_result = await verify_startup()
    if not verify_result["llm_active"]:
        logger.error("Startup verification failed: LLM not active")
        raise RuntimeError("LLM-mandatory requirement not met")
    
    logger.info(f"Startup verification passed: mode={verify_result['mode']}")
    
    # Initialize telemetry
    init_telemetry()
    
    yield
    
    # Shutdown
    logger.info("Shutting down NCERT AI Tutor backend")


app = FastAPI(
    title="NCERT AI Tutor API",
    description="Agentic AI Tutor grounded in NCERT textbooks (Phase 1)",
    version="0.11.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
if settings.rate_limit_enabled:
    app.add_middleware(RateLimitMiddleware, rpm=settings.rate_limit_rpm)

# Body size limit
app.add_middleware(BodySizeLimitMiddleware, max_size=settings.body_size_limit_mb * 1024 * 1024)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(mode.router, prefix="/mode", tags=["mode"])
app.include_router(agent.router, prefix="/agent", tags=["agent"])
app.include_router(ingest_pdf.router, prefix="/ingest", tags=["ingest"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(cache.router, prefix="/cache", tags=["cache"])
app.include_router(attempts.router, prefix="/attempts", tags=["attempts"])
app.include_router(exports.router, prefix="/exports", tags=["exports"])
app.include_router(textbooks.router, prefix="/textbooks", tags=["textbooks"])
app.include_router(multimodal.router, prefix="/multimodal", tags=["multimodal"])
app.include_router(visuals.router, prefix="/generate", tags=["visuals"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
app.include_router(provider_io.router, prefix="/provider-io", tags=["provider-io"])
# app.include_router(retrieval.router, prefix="/retrieval", tags=["retrieval"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NCERT AI Tutor",
        "version": "0.11.0",
        "phase": "1",
        "increment": "11",
        "status": "active"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.environment == "development"
    )
