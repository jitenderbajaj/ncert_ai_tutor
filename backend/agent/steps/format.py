# FILE: backend/agent/steps/format.py
"""
Format step: produce final response envelope
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def format_step(
    draft_answer: str,
    citations: List[str],
    engagement_meta: Dict[str, Any],
    governance_verdict: str,
    policy_messages: List[str],
    safety_meta: Dict[str, Any],
    correlation_id: str
) -> Dict[str, Any]:
    """
    Format final response envelope
    
    Returns:
        {answer: str, meta: Dict}
    """
    logger.debug(f"[{correlation_id}] Formatting final response")
    
    # Build final answer
    final_answer = draft_answer
    
    # Add engagement interventions inline
    for intervention in engagement_meta.get("interventions", []):
        if intervention["position"] == "inline":
            final_answer += f"\n\n💡 {intervention['content']}"
    
    # Add policy messages if needed
    if governance_verdict != "pass" and policy_messages:
        final_answer += f"\n\n⚠️ Note: {policy_messages[0]}"
    
    # Build meta
    meta = {
        "citations": citations,
        "governance_verdict": governance_verdict,
        "policy_messages": policy_messages,
        "safety_meta": safety_meta,
        "engagement_meta": engagement_meta
    }
    
    return {
        "answer": final_answer,
        "meta": meta
    }
