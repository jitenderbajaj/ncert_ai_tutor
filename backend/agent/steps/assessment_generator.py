# FILE: backend/agent/steps/assessment_generator.py (NEW FILE NEEDED)

"""
Assessment generation agent for book-level question papers
"""
import logging
from typing import Dict, Any, List, Optional

from backend.config import get_settings
from backend.services.multi_chapter_retrieval import retrieve_book_level
from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)
settings = get_settings()


def generate_question_paper(
    book_id: str,
    num_questions: int = 20,
    difficulty_mix: Dict[str, int] = None,
    bloom_mix: Dict[str, int] = None,
    include_hots: int = 5
) -> Dict[str, Any]:
    """
    Generate a question paper covering the entire book/subject.
    
    Process:
    1. Retrieve summaries from all chapters
    2. Analyze topic coverage
    3. Use LLM to generate balanced questions across chapters
    4. Tag questions with difficulty, Bloom level, HOTS
    
    Args:
        book_id: Book identifier
        num_questions: Total questions to generate (default: 20)
        difficulty_mix: {"easy": 8, "medium": 8, "hard": 4}
        bloom_mix: {"remember": 5, "understand": 5, "apply": 5, "analyze": 3, "evaluate": 2}
        include_hots: Number of HOTS questions
    
    Returns:
        {
            "questions": [...],
            "chapter_distribution": {...},
            "difficulty_distribution": {...},
            "metadata": {...}
        }
    """
    logger.info(f"[QUESTION PAPER] Generating {num_questions} questions for {book_id}")
    
    # Default mixes
    if difficulty_mix is None:
        difficulty_mix = {"easy": 8, "medium": 8, "hard": 4}
    
    if bloom_mix is None:
        bloom_mix = {
            "remember": 5,
            "understand": 5,
            "apply": 5,
            "analyze": 3,
            "evaluate": 2
        }
    
    # Step 1: Retrieve summaries from all chapters
    book_results = retrieve_book_level(
        query="main topics concepts learning objectives",
        book_id=book_id,
        top_k_per_chapter=1,  # Just the summary
        use_summary_index=True
    )
    
    chapters = book_results["chapters"]
    summaries_by_chapter = {}
    
    for chapter_id, results in book_results["results_by_chapter"].items():
        if results:
            summaries_by_chapter[chapter_id] = results[0]["text"]
    
    logger.info(f"[QUESTION PAPER] Retrieved summaries from {len(summaries_by_chapter)} chapters")
    
    # Step 2: Use LLM to generate questions
    registry = get_provider_registry()
    
    # Build prompt with chapter summaries
    summaries_text = "\n\n".join([
        f"**Chapter {ch}:**\n{summary[:500]}"
        for ch, summary in summaries_by_chapter.items()
    ])
    
    prompt = f"""You are an expert NCERT question paper creator. Generate {num_questions} questions covering all chapters of this book.

**Book:** {book_id}
**Chapters:** {", ".join(chapters)}

**Chapter Summaries:**
{summaries_text}

**Requirements:**
1. Total questions: {num_questions}
2. Difficulty distribution: {difficulty_mix}
3. Bloom level distribution: {bloom_mix}
4. Include {include_hots} HOTS (Higher Order Thinking Skills) questions
5. Cover ALL chapters proportionally
6. Mix question types: MCQ, Short Answer, Long Answer

**Question Paper (JSON format):**
Generate questions in this format:
{{
  "questions": [
    {{
      "question_id": "Q001",
      "chapter_id": "CH1",
      "question_text": "...",
      "question_type": "mcq",
      "choices": ["A", "B", "C", "D"],
      "correct_answer": "A",
      "difficulty": "easy",
      "bloom_level": "remember",
      "hots": false,
      "marks": 1
    }},
    ...
  ]
}}
"""
    
    try:
        response = registry.generate(
            prompt=prompt,
            temperature=0.3,  # Slight creativity for question variety
            max_tokens=4000,
            correlation_id=f"qpaper_{book_id}"
        )
        
        # Parse JSON response
        import json
        questions_data = json.loads(response["text"])
        
        logger.info(f"[QUESTION PAPER] ✅ Generated {len(questions_data['questions'])} questions")
        
        return {
            "questions": questions_data["questions"],
            "chapter_distribution": calculate_chapter_distribution(questions_data["questions"]),
            "difficulty_distribution": difficulty_mix,
            "bloom_distribution": bloom_mix,
            "metadata": {
                "book_id": book_id,
                "total_questions": num_questions,
                "chapters_covered": chapters,
                "generated_by": response.get("provider"),
                "model": response.get("model")
            }
        }
    
    except Exception as e:
        logger.error(f"[QUESTION PAPER] Failed: {e}")
        raise


def calculate_chapter_distribution(questions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate how many questions per chapter"""
    distribution = {}
    for q in questions:
        ch = q.get("chapter_id", "UNKNOWN")
        distribution[ch] = distribution.get(ch, 0) + 1
    return distribution
