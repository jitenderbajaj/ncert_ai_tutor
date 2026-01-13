# FILE: tests/conftest.py (COMPLETE UPDATED VERSION)

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from backend.config import get_settings

@pytest.fixture(scope="session")
def settings():
    """Provide settings for tests"""
    return get_settings()

@pytest.fixture
def test_data_dir():
    """Provide test data directory"""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_pdf_path(test_data_dir):
    """Provide sample PDF path"""
    return test_data_dir / "sample.pdf"

@pytest.fixture
def sample_question():
    """Sample question for testing"""
    return "What is photosynthesis?"

@pytest.fixture
def sample_context():
    """Sample context for testing"""
    return [
        {
            "text": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "score": 0.95
        }
    ]
