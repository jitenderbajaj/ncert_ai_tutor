# FILE: backend/agent/steps/syllabus.py
"""
Syllabus mapper step: tag with NCERT curriculum
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


async def syllabus_mapper_step(
    question: str,
    retrieve_results: List[Dict[str, Any]],
    book_id: str,
    chapter_id: str,
    correlation_id: str
) -> Dict[str, Any]:
    """
    Map question and results to NCERT syllabus tags
    
    Returns:
        {tags: List[str], confidence: float}
    """
    logger.debug(f"[{correlation_id}] Mapping to syllabus")
    
    # Stub: would use LLM or rule-based mapping
    tags = [f"{book_id}_{chapter_id}"]
    
    # Check if results are well-aligned
    if retrieve_results and len(retrieve_results) >= 3:
        confidence = 0.85
        tags.append("curriculum_aligned")
    else:
        confidence = 0.65
    
    return {
        "tags": tags,
        "confidence": confidence
    }
