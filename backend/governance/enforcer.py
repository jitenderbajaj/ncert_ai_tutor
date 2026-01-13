# FILE: backend/governance/enforcer.py
"""
Governance enforcer: apply coverage, safety, grounding checks
"""
import logging
from typing import Dict, Any, List

from backend.config import get_settings
from backend.governance.policy import load_governance_policy
from backend.governance.redaction import redact_pii

logger = logging.getLogger(__name__)
settings = get_settings()


def enforce_governance(
    answer: str,
    coverage: float,
    correlation_id: str
) -> Dict[str, Any]:
    """
    Enforce governance policy on answer
    
    Returns:
        {verdict: str, policy_messages: List[str]}
    """
    logger.debug(f"[{correlation_id}] Enforcing governance")
    
    policy = load_governance_policy(settings.governance_policy)
    
    verdict = "pass"
    policy_messages = []
    
    # Check coverage threshold
    threshold = policy.get("coverage_threshold", 0.6)
    if coverage < threshold:
        verdict = "coverage_warning"
        policy_messages.append(
            f"Answer may not fully cover the topic (coverage: {coverage:.2f})"
        )
    
    # Check for unsafe content (stub)
    if contains_unsafe_content(answer):
        verdict = "unsafe"
        policy_messages.append("Content flagged for safety review")
    
    # Redact PII if enabled
    if policy.get("redaction_pii", True) and settings.redaction_enabled:
        answer = redact_pii(answer)
    
    return {
        "verdict": verdict,
        "policy_messages": policy_messages,
        "redacted_answer": answer
    }


def contains_unsafe_content(text: str) -> bool:
    """Check if text contains unsafe content (stub)"""
    # Stub: would use content moderation API or rules
    unsafe_keywords = ["violence", "explicit"]
    return any(kw in text.lower() for kw in unsafe_keywords)
