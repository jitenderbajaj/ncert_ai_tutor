# FILE: backend/agent/steps/reflect.py
"""
Reflect step: assess answer quality and decide if refinement needed
Single retry limit enforced
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


async def reflect_step(
    question: str,
    retrieve_results: List[Dict[str, Any]],
    syllabus_tags: List[str],
    confidence: float,
    correlation_id: str
) -> Dict[str, Any]:
    """
    Reflect on retrieval quality and decide if refinement needed
    
    Returns:
        {needs_refinement: bool, reason: str, refined_query: str}
    """
    logger.debug(f"[{correlation_id}] Reflecting on retrieval quality")
    
    needs_refinement = False
    reason = ""
    refined_query = question
    
    # Check if confidence is low
    if confidence < 0.7:
        needs_refinement = True
        reason = "Low confidence in syllabus mapping"
        refined_query = f"Explain in detail: {question}"
    
    # Check if results are sparse
    elif len(retrieve_results) < 3:
        needs_refinement = True
        reason = "Insufficient retrieved passages"
        refined_query = f"{question} (include examples and explanations)"
    
    # Check if results have low scores
    elif retrieve_results and max(r.get("score", 0) for r in retrieve_results) < 0.5:
        needs_refinement = True
        reason = "Low retrieval scores"
        refined_query = f"What is {question}? Provide detailed explanation"
    
    return {
        "needs_refinement": needs_refinement,
        "reason": reason,
        "refined_query": refined_query
    }
