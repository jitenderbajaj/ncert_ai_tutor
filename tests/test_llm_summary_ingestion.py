# FILE: tests/test_llm_summary_ingestion.py
"""
Tests for LLM-based summary generation in dual-index ingestion

Tests:
1. LLM summary generation returns correct structure
2. Summary metadata is stored separately
3. Fallback mechanism when LLM unavailable
4. Summary chunking based on size
"""
import pytest
from pathlib import Path
from backend.services.ingestion_dual import (
    generate_chapter_summary_with_llm,
    create_summary_chunks
)


def test_llm_summary_generation(settings):
    """
    Test LLM summary generation returns proper structure and content.
    
    Verifies:
    - Returns dict with 'summary' and 'metadata' keys
    - Summary is a string with reasonable length
    - Metadata contains required fields
    """
    # Create sample chapter text (long enough to be meaningful)
    sample_text = """
    Photosynthesis is a fundamental process in biology where plants, algae, and some bacteria 
    convert light energy into chemical energy stored in glucose. This process occurs primarily 
    in chloroplasts, which contain chlorophyll pigments that absorb light energy.
    
    The process can be divided into two main stages: the light-dependent reactions and the 
    light-independent reactions (Calvin cycle). During light-dependent reactions, light energy 
    is converted into ATP and NADPH, while oxygen is released as a byproduct.
    
    In the light-independent reactions, carbon dioxide is fixed into organic molecules using 
    the energy from ATP and NADPH. This results in the production of glucose and other 
    carbohydrates that serve as energy sources for the plant and other organisms.
    
    Photosynthesis is essential for life on Earth as it produces oxygen and organic compounds 
    that support the food chain. Without photosynthesis, most life forms would not be able 
    to survive, making it one of the most important biological processes.
    """ * 10  # Repeat to make it substantial
    
    # Generate summary
    result = generate_chapter_summary_with_llm(
        chapter_text=sample_text,
        book_id="TEST",
        chapter_id="CH1",
        seed=42
    )
    
    # Verify structure
    assert isinstance(result, dict), "Result should be a dict"
    assert "summary" in result, "Result should have 'summary' key"
    assert "metadata" in result, "Result should have 'metadata' key"
    
    # Verify summary content
    summary_text = result["summary"]
    assert isinstance(summary_text, str), "Summary should be a string"
    assert len(summary_text) > 100, f"Summary too short: {len(summary_text)} chars (expected > 100)"
    assert "photosynthesis" in summary_text.lower() or "light" in summary_text.lower(), \
        "Summary should contain relevant content"
    
    # Verify metadata
    metadata = result["metadata"]
    assert metadata["book_id"] == "TEST"
    assert metadata["chapter_id"] == "CH1"
    assert metadata["seed"] == 42
    assert "generated_by" in metadata
    assert "timestamp" in metadata
    assert "input_chars" in metadata
    assert "output_chars" in metadata
    assert "compression_ratio" in metadata
    assert metadata["generation_status"] in ["success", "fallback"]


def test_summary_metadata_stored_separately(settings, tmp_path):
    """
    Test that summary metadata is stored in separate file.
    
    Verifies:
    - Metadata file is created
    - Contains all required fields
    - Can be read back correctly
    """
    from backend.services.ingestion_dual import save_summary_metadata
    
    # Create test metadata
    metadata = {
        "chapter_id": "CH1",
        "book_id": "TEST",
        "generated_by": "test/model",
        "provider": "test",
        "model": "model",
        "seed": 42,
        "timestamp": "2025-11-16T12:00:00",
        "input_chars": 5000,
        "output_chars": 1000,
        "compression_ratio": 0.2,
        "generation_status": "success"
    }
    
    # Override shards_dir for testing
    import backend.services.ingestion_dual as ingestion_module
    original_settings = ingestion_module.settings
    
    # Create a mock settings object
    class MockSettings:
        shards_dir = str(tmp_path)
    
    ingestion_module.settings = MockSettings()
    
    try:
        # Save metadata
        save_summary_metadata(metadata, "TEST", "CH1")
        
        # Verify file exists
        metadata_file = tmp_path / "TEST_CH1" / "summary_metadata.json"
        assert metadata_file.exists(), "Metadata file should be created"
        
        # Read and verify content
        import json
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata["chapter_id"] == "CH1"
        assert loaded_metadata["book_id"] == "TEST"
        assert loaded_metadata["generated_by"] == "test/model"
        assert loaded_metadata["seed"] == 42
    
    finally:
        # Restore original settings
        ingestion_module.settings = original_settings


@pytest.mark.skip(reason="Requires LLM provider configuration")
def test_fallback_when_llm_unavailable(settings):
    """
    Test fallback mechanism when LLM is unavailable.
    
    Verifies:
    - Returns fallback summary
    - Metadata indicates fallback status
    - No crash or exception
    """
    sample_text = "Short sample text for fallback test." * 50
    
    # This would need to mock LLM failure
    # Skipped for now as it requires provider setup
    pass


def test_summary_chunking(settings):
    """
    Test summary chunking based on size threshold.
    
    Verifies:
    - Small summaries remain single chunk
    - Large summaries split appropriately
    - All chunks have required metadata
    """
    # Test 1: Small summary (single chunk)
    small_summary = "This is a short summary." * 10
    
    chunks_small = create_summary_chunks(
        summary_text=small_summary,
        book_id="TEST",
        chapter_id="CH1",
        max_chunk_size=8000
    )
    
    assert len(chunks_small) == 1, "Small summary should be single chunk"
    assert chunks_small[0]["metadata"]["chunk_strategy"] == "single"
    assert chunks_small[0]["metadata"]["total_sections"] == 1
    
    # Test 2: Large summary (multiple chunks)
    paragraphs = [
        "This is paragraph one discussing photosynthesis in detail. " * 50,
        "This is paragraph two covering chloroplast structure. " * 50,
        "This is paragraph three explaining light reactions. " * 50,
        "This is paragraph four describing the Calvin cycle. " * 50
    ]
    large_summary = "\n\n".join(paragraphs)
    
    chunks_large = create_summary_chunks(
        summary_text=large_summary,
        book_id="TEST",
        chapter_id="CH1",
        max_chunk_size=5000  # Force splitting
    )
    
    assert len(chunks_large) > 1, f"Large summary should split, got {len(chunks_large)} chunks"
    
    # Verify all chunks have total_sections
    for chunk in chunks_large:
        assert "total_sections" in chunk["metadata"]
        assert chunk["metadata"]["total_sections"] == len(chunks_large)
        assert chunk["metadata"]["chunk_strategy"] == "split"


def test_summary_content_quality(settings):
    """
    Test that generated summary contains relevant content.
    
    Verifies:
    - Summary is not empty
    - Summary is shorter than input
    - Summary contains key terms from input
    """
    sample_text = """
    Chemical reactions are processes where substances transform into new substances.
    These reactions involve breaking and forming chemical bonds. The law of conservation
    of mass states that matter cannot be created or destroyed in a chemical reaction.
    
    There are several types of chemical reactions including combination reactions,
    decomposition reactions, displacement reactions, and double displacement reactions.
    Each type has distinct characteristics and follows specific patterns.
    
    Oxidation and reduction are important concepts in chemistry. Oxidation involves
    loss of electrons or gain of oxygen, while reduction involves gain of electrons
    or loss of oxygen. These processes often occur together in redox reactions.
    """ * 5
    
    result = generate_chapter_summary_with_llm(
        chapter_text=sample_text,
        book_id="TEST",
        chapter_id="CH1",
        seed=42
    )
    
    summary = result["summary"]
    
    # Verify compression
    assert len(summary) < len(sample_text), "Summary should be shorter than input"
    
    # Verify it's not too short (should have substance)
    assert len(summary) > 200, "Summary should have reasonable length"
    
    # Verify key terms appear (case insensitive check)
    summary_lower = summary.lower()
    key_terms = ["chemical", "reaction", "oxidation", "reduction"]
    found_terms = sum(1 for term in key_terms if term in summary_lower)
    
    assert found_terms >= 2, f"Summary should contain key terms, found {found_terms}/4"
