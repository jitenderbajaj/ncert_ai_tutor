# FILE: backend/models/tools.py
"""
Tool schema models
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """Tool schema definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class ToolCall(BaseModel):
    """Tool call record"""
    tool_name: str
    correlation_id: str
    timestamp: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class RegisteredTools(BaseModel):
    """Registered tools response"""
    tools: List[ToolSchema]


# Tool schemas
TOOL_SCHEMAS = {
    "retrieve_detail": ToolSchema(
        name="retrieve_detail",
        description="Retrieve detailed passages from chapter",
        parameters={
            "query": {"type": "string", "required": True},
            "book_id": {"type": "string", "required": True},
            "chapter_id": {"type": "string", "required": True},
            "top_k": {"type": "integer", "default": 5}
        },
        returns={
            "results": {"type": "array", "items": {"type": "object"}}
        }
    ),
    "retrieve_summary": ToolSchema(
        name="retrieve_summary",
        description="Retrieve summary passages from chapter",
        parameters={
            "query": {"type": "string", "required": True},
            "book_id": {"type": "string", "required": True},
            "chapter_id": {"type": "string", "required": True},
            "top_k": {"type": "integer", "default": 3}
        },
        returns={
            "results": {"type": "array", "items": {"type": "object"}}
        }
    ),
    "query_transform": ToolSchema(
        name="query_transform",
        description="Transform user query for better retrieval",
        parameters={
            "query": {"type": "string", "required": True},
            "context": {"type": "string", "required": False}
        },
        returns={
            "transformed_query": {"type": "string"}
        }
    ),
    "calculator": ToolSchema(
        name="calculator",
        description="Perform mathematical calculations",
        parameters={
            "expression": {"type": "string", "required": True}
        },
        returns={
            "result": {"type": "number"}
        }
    ),
    "generate_image": ToolSchema(
        name="generate_image",
        description="Generate image from text prompt",
        parameters={
            "prompt": {"type": "string", "required": True},
            "seed": {"type": "integer", "default": 42},
            "width": {"type": "integer", "default": 512},
            "height": {"type": "integer", "default": 512}
        },
        returns={
            "image_path": {"type": "string"},
            "metadata": {"type": "object"}
        }
    ),
    "generate_diagram": ToolSchema(
        name="generate_diagram",
        description="Generate diagram from description",
        parameters={
            "prompt": {"type": "string", "required": True},
            "format": {"type": "string", "enum": ["mermaid", "graphviz", "ascii"], "default": "mermaid"}
        },
        returns={
            "diagram": {"type": "string"},
            "format": {"type": "string"}
        }
    )
}
