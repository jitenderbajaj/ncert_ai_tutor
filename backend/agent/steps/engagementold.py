# FILE: backend/agent/steps/engagement.py
"""
Engagement step: add beyond-textbook engagement
"""
import logging
from typing import Dict, Any, Optional

from backend.agent.engagement import generate_engagement_interventions

logger = logging.getLogger(__name__)


def engagement_step(
    draft_answer: str,
    question: str,
    hots_level: Optional[str],
    correlation_id: str
) -> Dict[str, Any]:
    """
    Add engagement interventions to answer
    
    Returns:
        {interventions: List[Dict], boredom_meta: Dict}
    """
    logger.debug(f"[{correlation_id}] Adding engagement interventions")
    
    # Stub: would check user history for boredom
    boredom_detected = False
    
    interventions = generate_engagement_interventions(
        draft_answer=draft_answer,
        question=question,
        boredom_detected=boredom_detected,
        hots_level=hots_level
    )
    
    return {
        "interventions": interventions,
        "boredom_meta": {
            "detected": boredom_detected,
            "score": 0.0
        }
    }
