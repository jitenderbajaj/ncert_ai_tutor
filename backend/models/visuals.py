# FILE: backend/models/visuals.py
"""
Visual generation models
"""
from typing import Optional
from pydantic import BaseModel, Field


class GenerateImageRequest(BaseModel):
    """Generate image request"""
    prompt: str
    seed: Optional[int] = 42
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=50, ge=10, le=100)
    provider: str = Field(default="local", description="local or online")


class GenerateDiagramRequest(BaseModel):
    """Generate diagram request"""
    prompt: str
    format: str = Field(default="mermaid", description="mermaid, graphviz, or ascii")
