# FILE: backend/agent/steps/engagement.py
"""
Engagement step: add beyond-textbook engagement (LLM-powered)

Note:
- For TOC / structure-lookup flows, Compose already ends with
  "Would you like to explore it further?".
  Adding micro-quizzes/choose-path can feel like a second CTA, so we skip.
"""
import logging
from typing import Dict, Any, Optional, List

from backend.agent.engagement import generate_engagement_interventions

logger = logging.getLogger(__name__)


def _is_toc_sources(sources: Optional[List[Dict[str, Any]]]) -> bool:
    """
    Detect TOC/structure retrieval outputs.
    retriever.py sets: metadata.type = "structure" for structurelookup results. [file:16]
    """
    for d in (sources or []):
        if not isinstance(d, dict):
            continue
        meta = d.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("type") == "structure":
            return True
    return False


def engagement_step(
    draft_answer: str,
    question: str,
    chat_history: Optional[List[Dict[str, str]]],
    sources: Optional[List[Dict[str, Any]]],
    hots_level: Optional[str],
    correlation_id: str,
) -> Dict[str, Any]:
    """
    Returns a dict payload:
    {
      "interventions": [{"position":"inline","type":"...","content":"..."}],
      "boredom_meta": {"detected": bool, "score": float, "reasons": [...]},
      "provider_meta": {"provider": str, "model": str, ...}
    }
    """
    logger.debug(f"[{correlation_id}] Adding engagement interventions")

    # Skip engagement for TOC/structure flows to avoid confusing double CTAs. [file:16]
    if _is_toc_sources(sources):
        return {
            "interventions": [],  # format_step will append nothing. [file:14]
            "boredom_meta": {"detected": False, "score": 0.0, "reasons": ["toc_skip"]},
            "provider_meta": {
                "provider": "skipped",
                "model": "none",
                "router_reason": "toc_skip",
                "duration_ms": 0,
                "mode": "n/a",
            },
        }

    return generate_engagement_interventions(
        draft_answer=draft_answer,
        question=question,
        chat_history=chat_history or [],
        sources=sources or [],
        hots_level=hots_level,
        correlation_id=correlation_id,
    )
