# FILE: backend/governance/audit.py
"""
Audit logging for governance decisions
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def log_governance_decision(
    correlation_id: str,
    verdict: str,
    policy_messages: List[str],
    coverage: float,
    answer_hash: str
) -> None:
    """Log governance decision to audit trail"""
    audit_dir = Path(settings.logs_dir) / "governance"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    audit_file = audit_dir / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "correlation_id": correlation_id,
        "verdict": verdict,
        "policy_messages": policy_messages,
        "coverage": coverage,
        "answer_hash": answer_hash
    }
    
    with open(audit_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')
