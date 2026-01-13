# FILE: backend/routes/visuals.py
"""
Visual generation endpoints
"""
import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.multimodal.image_generate import generate_image
from backend.multimodal.diagram_generate import generate_diagram

logger = logging.getLogger(__name__)
router = APIRouter()


class GenerateImageRequest(BaseModel):
    """Generate image request"""
    prompt: str
    seed: Optional[int] = 42
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=50, ge=10, le=100)
    provider: str = Field(default="local")


class GenerateDiagramRequest(BaseModel):
    """Generate diagram request"""
    prompt: str
    format: str = Field(default="mermaid")


@router.post("/image")
async def generate_image_endpoint(request: GenerateImageRequest):
    """Generate image"""
    logger.info(f"Generate image: {request.prompt[:50]}")
    
    result = generate_image(
        prompt=request.prompt,
        seed=request.seed,
        width=request.width,
        height=request.height,
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.num_inference_steps,
        provider=request.provider
    )
    
    return result


@router.post("/diagram")
async def generate_diagram_endpoint(request: GenerateDiagramRequest):
    """Generate diagram"""
    logger.info(f"Generate diagram: {request.prompt[:50]}")
    
    result = generate_diagram(
        prompt=request.prompt,
        format=request.format
    )
    
    return result
