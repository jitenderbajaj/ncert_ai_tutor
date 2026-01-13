# FILE: backend/agent/engagement.py
"""
Engagement coordinator for proactive tutor behavior
Increment 11: boredom detection, HOTS controls, beyond-textbook hooks
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def detect_boredom_signals(user_id: str, recent_interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect boredom from interaction patterns"""
    # Stub: would analyze recent_interactions for patterns
    # - Repeated short responses
    # - Long gaps between interactions
    # - Declining engagement metrics
    
    boredom_score = 0.0
    signals = []
    
    if len(recent_interactions) > 5:
        # Check for short responses
        short_responses = sum(1 for i in recent_interactions if len(i.get("response", "")) < 20)
        if short_responses > len(recent_interactions) * 0.6:
            boredom_score += 0.3
            signals.append("short_responses")
    
    return {
        "boredom_score": boredom_score,
        "signals": signals,
        "threshold": 0.5,
        "intervention_needed": boredom_score > 0.5
    }


def generate_engagement_interventions(
    draft_answer: str,
    question: str,
    boredom_detected: bool,
    hots_level: Optional[str]
) -> List[Dict[str, Any]]:
    """Generate engagement interventions"""
    interventions = []
    
    # Analogies
    interventions.append({
        "type": "analogy",
        "content": "Think of it like...",
        "position": "inline"
    })
    
    # Real-world hook
    interventions.append({
        "type": "real_world",
        "content": "This concept is used in everyday life when...",
        "position": "inline"
    })
    
    # Micro-quiz
    if not boredom_detected:
        interventions.append({
            "type": "micro_quiz",
            "content": "Quick check: Can you explain why...?",
            "position": "end"
        })
    
    # Progressive hints
    if hots_level == "hard":
        interventions.append({
            "type": "hint",
            "content": "Hint: Consider the relationship between...",
            "position": "inline"
        })
    
    return interventions
