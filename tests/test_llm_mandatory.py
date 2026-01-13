# FILE: tests/test_llm_mandatory.py
"""Test LLM-mandatory requirement"""
import pytest

def test_llm_providers_available(settings):
    """Test that LLM providers are available"""
    from backend.providers.registry import get_provider_registry
    
    registry = get_provider_registry()
    assert len(registry.providers) > 0, "No LLM providers available"

def test_composition_requires_llm(sample_question):
    """Test that composition always invokes LLM"""
    from backend.agent.steps.compose import compose_step
    
    # This should not raise - LLM must be available
    result = compose_step(
        question=sample_question,
        retrieve_results=[{"id": "test", "text": "Test passage"}],
        syllabus_tags=["test"],
        book_id="TEST",
        chapter_id="TEST",
        correlation_id="test"
    )
    
    assert "answer" in result
    assert "citations" in result
