# FILE: backend/agent/steps/governance.py
"""
Governance step: apply safety and coverage checks
"""
import logging
from typing import Dict, Any, List

from backend.governance.enforcer import enforce_governance
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def governance_step(
    draft_answer: str,
    citations: List[str],
    retrieve_results: List[Dict[str, Any]],
    correlation_id: str
) -> Dict[str, Any]:
    """
    Apply governance: coverage, safety, grounding
    
    Returns:
        {verdict: str, policy_messages: List[str], safety_meta: Dict}
    """
    logger.debug(f"[{correlation_id}] Applying governance")
    
    # Calculate coverage
    coverage = len(citations) / max(len(retrieve_results), 1) if retrieve_results else 0.0
    
    # Enforce governance
    result = enforce_governance(
        answer=draft_answer,
        coverage=coverage,
        correlation_id=correlation_id
    )
    
    verdict = result["verdict"]
    policy_messages = result["policy_messages"]
    safety_meta = {
        "coverage": coverage,
        "threshold": settings.coverage_threshold,
        "meets_threshold": coverage >= settings.coverage_threshold
    }
    
    return {
        "verdict": verdict,
        "policy_messages": policy_messages,
        "safety_meta": safety_meta
    }
