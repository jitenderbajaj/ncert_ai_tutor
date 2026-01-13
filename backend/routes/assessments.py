# FILE: backend/routes/assessments.py (ADD NEW ENDPOINT)

@router.post("/generate/question-paper")
async def generate_question_paper_endpoint(
    book_id: str,
    num_questions: int = 20,
    difficulty_mix: Optional[Dict[str, int]] = None,
    bloom_mix: Optional[Dict[str, int]] = None,
    include_hots: int = 5
):
    """
    Generate a question paper covering the entire book/subject.
    
    Example request:
    POST /assessments/generate/question-paper
    {
        "book_id": "BOOK123",
        "num_questions": 20,
        "difficulty_mix": {"easy": 8, "medium": 8, "hard": 4},
        "include_hots": 5
    }
    """
    from backend.agent.steps.assessment_generator import generate_question_paper
    
    try:
        result = generate_question_paper(
            book_id=book_id,
            num_questions=num_questions,
            difficulty_mix=difficulty_mix,
            bloom_mix=bloom_mix,
            include_hots=include_hots
        )
        
        return {
            "status": "success",
            **result
        }
    
    except Exception as e:
        logger.error(f"Question paper generation failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
