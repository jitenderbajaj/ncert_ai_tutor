# FILE: backend/services/multi_chapter_retrieval.py (NEW FILE NEEDED)

"""
Multi-chapter retrieval for book-level and subject-level queries
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.config import get_settings
from backend.services.retrieval_dual import retrieve_passages

logger = logging.getLogger(__name__)
settings = get_settings()


def retrieve_book_level(
    query: str,
    book_id: str,
    top_k_per_chapter: int = 3,
    use_summary_index: bool = True
) -> Dict[str, Any]:
    """
    Retrieve across all chapters in a book.
    
    Strategy:
    1. List all chapters for book_id
    2. Query summary index for each chapter (LLM-generated summaries)
    3. Aggregate and rank results across chapters
    4. Return chapter-grouped results
    
    Use case: "What topics are covered in this book?"
              "Prepare a question paper for Chemistry class 10"
    
    Args:
        query: Search query
        book_id: Book identifier (e.g., "BOOK123")
        top_k_per_chapter: Results per chapter
        use_summary_index: If True, query summary index (faster, high-level)
    
    Returns:
        {
            "chapters": ["CH1", "CH2", ...],
            "results_by_chapter": {"CH1": [...], "CH2": [...]},
            "aggregated_results": [...],
            "topic_coverage": {...}
        }
    """
    logger.info(f"[BOOK RETRIEVE] Query: '{query}' for book {book_id}")
    
    # List all chapters for this book
    chapters = list_chapters_for_book(book_id)
    logger.info(f"[BOOK RETRIEVE] Found {len(chapters)} chapters for {book_id}")
    
    if not chapters:
        logger.warning(f"No chapters found for book {book_id}")
        return {
            "chapters": [],
            "results_by_chapter": {},
            "aggregated_results": [],
            "topic_coverage": {}
        }
    
    # Query each chapter
    index_type = "summary" if use_summary_index else "detail"
    results_by_chapter = {}
    all_results = []
    
    for chapter_id in chapters:
        logger.debug(f"[BOOK RETRIEVE] Querying {book_id}/{chapter_id} ({index_type} index)")
        
        try:
            chapter_results = retrieve_passages(
                query=query,
                book_id=book_id,
                chapter_id=chapter_id,
                index_type=index_type,
                top_k=top_k_per_chapter,
                expand_to_parents=False  # Use summaries directly
            )
            
            results_by_chapter[chapter_id] = chapter_results
            
            # Add chapter context to each result
            for result in chapter_results:
                result["chapter_id"] = chapter_id
                result["book_id"] = book_id
                all_results.append(result)
        
        except Exception as e:
            logger.warning(f"Failed to query {chapter_id}: {e}")
            results_by_chapter[chapter_id] = []
    
    # Sort all results by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Analyze topic coverage
    topic_coverage = analyze_topic_coverage(all_results)
    
    logger.info(f"[BOOK RETRIEVE] ✅ Retrieved {len(all_results)} results across {len(chapters)} chapters")
    
    return {
        "chapters": chapters,
        "results_by_chapter": results_by_chapter,
        "aggregated_results": all_results[:top_k_per_chapter * 3],  # Top results
        "topic_coverage": topic_coverage
    }


def list_chapters_for_book(book_id: str) -> List[str]:
    """
    List all chapters for a book by scanning shard directory.
    
    Args:
        book_id: Book identifier
    
    Returns:
        List of chapter IDs (sorted)
    """
    shards_base = Path(settings.shards_dir)
    chapters = []
    
    # Scan for directories matching {book_id}_{chapter_id}
    for shard_dir in shards_base.iterdir():
        if shard_dir.is_dir() and shard_dir.name.startswith(f"{book_id}_"):
            chapter_id = shard_dir.name.replace(f"{book_id}_", "")
            chapters.append(chapter_id)
    
    chapters.sort()  # Deterministic ordering
    
    return chapters


def analyze_topic_coverage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze topic coverage across retrieved results.
    
    Returns:
        {
            "chapters_covered": ["CH1", "CH2", ...],
            "topics": ["photosynthesis", "respiration", ...],
            "chapter_weights": {"CH1": 0.3, "CH2": 0.5, ...}
        }
    """
    chapters_covered = list(set(r.get("chapter_id") for r in results if r.get("chapter_id")))
    
    # Simple keyword extraction (can be enhanced with NLP)
    topics = []
    for result in results[:10]:  # Top 10 for topic extraction
        text = result.get("text", "").lower()
        # Extract noun phrases (simplified)
        words = text.split()
        topics.extend([w for w in words if len(w) > 5])
    
    # Count chapter representation
    chapter_counts = {}
    total_score = sum(r.get("score", 0) for r in results)
    
    for result in results:
        chapter_id = result.get("chapter_id")
        score = result.get("score", 0)
        if chapter_id:
            chapter_counts[chapter_id] = chapter_counts.get(chapter_id, 0) + score
    
    # Normalize to weights
    chapter_weights = {
        ch: count / total_score if total_score > 0 else 0
        for ch, count in chapter_counts.items()
    }
    
    return {
        "chapters_covered": sorted(chapters_covered),
        "topics": list(set(topics))[:20],  # Top 20 unique topics
        "chapter_weights": chapter_weights
    }
