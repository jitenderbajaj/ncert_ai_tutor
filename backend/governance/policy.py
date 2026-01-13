# FILE: backend/governance/policy.py
"""
Governance policy definitions
"""
from typing import Dict, Any


def load_governance_policy(policy_path: str) -> Dict[str, Any]:
    """Load governance policy from JSON file"""
    import json
    try:
        with open(policy_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default policy
        return {
            "coverage_threshold": 0.6,
            "safety_checks": ["grounding", "age_appropriate"],
            "refusal_triggers": ["out_of_scope", "unsafe"],
            "redaction_pii": True
        }
