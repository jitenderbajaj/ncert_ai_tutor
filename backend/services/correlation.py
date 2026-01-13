# FILE: backend/services/correlation.py
"""
Correlation ID utilities
"""
import uuid


def generate_correlation_id() -> str:
    """Generate unique correlation ID"""
    return str(uuid.uuid4())


def get_correlation_id() -> str:
    """Get current correlation ID (stub for context)"""
    return generate_correlation_id()
