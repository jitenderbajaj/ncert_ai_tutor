# FILE: tests/test_summary_chunking_threshold.py

import pytest
from backend.services.ingestion_dual import create_summary_chunks

def test_small_summary_single_chunk():
    """Test that small summaries stay as single chunk"""
    summary = "This is a short summary. " * 100  # ~2500 chars
    
    chunks = create_summary_chunks(
        summary_text=summary,
        book_id="TEST",
        chapter_id="CH1",
        max_chunk_size=8000
    )
    
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_strategy"] == "single"


def test_medium_summary_single_chunk():
    """Test that medium summaries stay as single chunk with 8000 threshold"""
    summary = "This is a medium summary. " * 300  # ~7500 chars
    
    chunks = create_summary_chunks(
        summary_text=summary,
        book_id="TEST",
        chapter_id="CH1",
        max_chunk_size=8000
    )
    
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_strategy"] == "single"


def test_large_summary_split():
    """Test that very large summaries get split"""
    summary = "Section 1\n\n" + ("Content. " * 1000) + "\n\nSection 2\n\n" + ("More content. " * 1000)
    # ~20000 chars
    
    chunks = create_summary_chunks(
        summary_text=summary,
        book_id="TEST",
        chapter_id="CH1",
        max_chunk_size=8000
    )
    
    assert len(chunks) > 1
    assert chunks[0]["metadata"]["chunk_strategy"] == "split"
    assert chunks[0]["metadata"]["total_sections"] == len(chunks)


def test_configurable_threshold():
    """Test that threshold is configurable"""
    summary = "Content. " * 1500  # ~12000 chars
    
    # With 8000 threshold: should split
    chunks_8k = create_summary_chunks(
        summary_text=summary,
        book_id="TEST",
        chapter_id="CH1",
        max_chunk_size=8000
    )
    assert len(chunks_8k) > 1
    
    # With 16000 threshold: should stay single
    chunks_16k = create_summary_chunks(
        summary_text=summary,
        book_id="TEST",
        chapter_id="CH1",
        max_chunk_size=16000
    )
    assert len(chunks_16k) == 1
