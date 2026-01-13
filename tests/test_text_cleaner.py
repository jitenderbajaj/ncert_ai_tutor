# FILE: tests/test_text_cleaner.py

import pytest
from backend.services.text_cleaner import clean_text_for_llm_summary

def test_remove_metadata():
    """Test metadata removal"""
    text = "CHAP 1.pmd CHAPTER 1 Content here Reprint 2025-26"
    cleaned = clean_text_for_llm_summary(text)
    
    assert "CHAP 1.pmd" not in cleaned
    assert "Reprint 2025-26" not in cleaned
    assert "CHAPTER 1" in cleaned
    assert "Content here" in cleaned


def test_remove_artifacts():
    """Test artifact removal"""
    text = "This is ished a test shed with artifacts be re"
    cleaned = clean_text_for_llm_summary(text)
    
    assert "ished" not in cleaned
    assert "shed" not in cleaned
    assert "be re" not in cleaned
    assert "This is a test with artifacts" in cleaned


def test_aggressive_mode():
    """Test aggressive cleaning removes Activities and Questions"""
    text = """
    CHAPTER 1
    Core content here.
    Activity 1.1
    Do this experiment.
    QUESTIONS
    1. What is X?
    2. What is Y?
    """
    
    cleaned = clean_text_for_llm_summary(text, aggressive=True)
    
    assert "CHAPTER 1" in cleaned
    assert "Core content" in cleaned
    assert "Activity 1.1" not in cleaned
    assert "QUESTIONS" not in cleaned
