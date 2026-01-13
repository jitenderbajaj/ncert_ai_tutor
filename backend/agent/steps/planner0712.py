# FILE: backend/agent/steps/planner.py
"""
Planner agent step - intelligent query routing and strategy selection

Responsibilities:
1. Analyze question intent and scope (chapter, book, subject)
2. Select appropriate index (detail vs summary)
3. Determine retrieval strategy (single-chapter, multi-chapter, assessment generation)
4. Set top_k and other retrieval parameters
5. Enable summary sampler for comprehensive queries

Routing Logic:
- "Explain photosynthesis" → detail index, single chapter
- "Summarize Chapter 3" → summary index, single chapter  
- "What topics are covered in this book?" → summary index, all chapters
- "Prepare a question paper for entire subject" → assessment generation, all chapters
"""
import logging
from typing import Dict, Any, Optional, List
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import Tuple # Ensure Tuple is imported
from backend.providers.registry import get_provider_registry


from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# --- START INCREMENT 11.2: HYBRID SEMANTIC ROUTER ---
_router_instance = None

class SemanticRouter:
    """
    Hybrid Intent Classifier using Embeddings (Fast) + LLM Fallback (Smart).
    """
    def __init__(self):
        logger.info(f"Loading Router Model: {settings.EMBEDDING_MODEL_NAME}")
        # Load same model as ingestion/retrieval to share memory
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        
        # Define Anchor Phrases for each Intent
        self.routes = {
            # "structure_lookup": [
            #     "table of contents", "list of topics", "what chapters are there", 
            #     "syllabus", "structure of the book", "what is covered", "list headings"
            # ],
            "structure_lookup": [
                "table of contents", "list of topics", "syllabus", "structure of the book",
                "explain a key concept", "what is the main idea", "teach me something new", 
                "overview of chapter"
            ],
            "book_aggregate": [
                "summarize the whole book", "overview of the text", "main themes of the book",
                "summary of all chapters", "brief overview of the subject"
            ],
            "generate_assessment": [
                "create a quiz", "generate questions", "prepare a test", "exam preparation",
                "make a question paper", "practice problems"
            ],
            "general_chat": [
                "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
                "how are you", "thanks", "thank you", "who are you", "what can you do"
            ],
            "detail": [
                "what is the definition", "explain the process", "how does it work",
                "detailed explanation", "specific facts about"
            ]
        }
        self.route_embeddings = {r: self.model.encode(p, convert_to_numpy=True) for r, p in self.routes.items()}

    def classify(self, query: str) -> Tuple[str, float]:
        query_emb = self.model.encode(query, convert_to_numpy=True)
        best_score = -1.0
        best_intent = "detail"
        for route, embeddings in self.route_embeddings.items():
            # util.cos_sim returns a PyTorch Tensor
            scores = util.cos_sim(query_emb, embeddings)[0]
            # Get max value from Tensor efficiently
            max_route_score = float(scores.max()) 
            
            if max_route_score > best_score:
                best_score = max_route_score
                best_intent = route
        return best_intent, best_score


def get_router():
    global _router_instance
    if _router_instance is None:
        _router_instance = SemanticRouter()
    return _router_instance

async def _llm_fallback_classification(question: str) -> str:
    """Tier 2: Ask the Local LLM to classify ambiguous queries."""
    from backend.providers.registry import get_provider_registry
    
    prompt = f"""Classify the user query into exactly one category:
    1. 'structure_lookup': Questions about table of contents, topics, hierarchy.
    2. 'book_aggregate': Requests for high-level summaries of the entire book.
    3. 'generate_assessment': Requests for quizzes, tests, exams.
    4. 'detail': Specific questions about concepts, definitions, or facts.
    5. 'general_chat': Conversational, out-of-scope questions, Greetings, "hi", "hello", "thanks", small talk.
    Query: "{question}"
    Return ONLY the category name."""
    # prompt = f"""
    # Classify the query into one category:
    
    # CATEGORIES:
    # 1. general_chat (Greetings, "hi", "hello", "thanks", small talk)
    # 2. structure_lookup (TOC, "what chapters", "list topics")
    # 3. detail (Questions about book content, facts, concepts)
    
    # Query: "{question}"
    # RESPONSE FORMAT: Just the category name.
    # """
    try:
        registry = get_provider_registry()
        # Sync call, returns dict
        result = registry.generate(prompt, temperature=0.0, max_tokens=15)
        # Extract content
        # response = result.get("content", "")
        response = result.get("text", "").strip().lower() 
        
        cleaned = response.strip().lower().replace("'", "").replace('"', "")
        for valid in ["structure_lookup", "book_aggregate", "generate_assessment", "detail", "general_chat"]:
            if valid in cleaned: return valid
        return "detail"
    except Exception as e:
        logger.error(f"LLM Fallback failed: {e}")
        return "detail"
# --- END INCREMENT 11.2 ROUTER ---

async def plan_step(
    question: str,
    book_id: str,
    chapter_id: Optional[str] = None,
    user_preferences: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Plan retrieval and composition strategy based on question analysis.
    
    Args:
        question: User's question
        book_id: Book identifier
        chapter_id: Chapter identifier (None for book-level queries)
        user_preferences: Optional user preferences (HOTS level, learning style, etc.)
    
    Returns:
        {
            "strategy": str,          # retrieve, book_aggregate, generate_assessment, general_chat
            "scope": str,             # chapter, book, subject, general
            "index_hint": str,        # detail, summary, none
            "top_k": int,             # Results per query
            "top_k_per_chapter": int, # For multi-chapter
            "summary_sampler": str,   # chapter, all
            "expand_to_parents": bool,# Enable parent expansion
            "enable_reflection": bool,# Enable reflection retry
            "assessment_params": dict,# For assessment generation
            "reasoning": str          # Why this plan was chosen
        }
    """
    logger.info(f"[PLANNER] Planning for: '{question[:60]}...'")
    logger.debug(f"[PLANNER] book_id={book_id}, chapter_id={chapter_id}")
    
    question_lower = question.lower().strip()
    
    # --- INCREMENT 11.2: HYBRID ROUTER LOGIC ---
    # 1. Semantic Routing (Tier 1)
    router = get_router()
    intent, confidence = router.classify(question)
    logger.info(f"[PLANNER] Router Prediction: {intent} (Confidence: {confidence:.2f})")
    
    # 2. Hybrid Fallback (Tier 2)
    if confidence < 0.60:
        logger.info("[PLANNER] Confidence low. Triggering LLM Fallback...")
        intent = await _llm_fallback_classification(question)
        logger.info(f"[PLANNER] LLM Fallback Verdict: {intent}")

    # 3. Construct Plan Directly (Replacing legacy build_plan)
    strategy = "retrieve"  # Default
    scope = "chapter" if chapter_id else "book"
    index_hint = "detail"
    summary_sampler = "none"
    assessment_params = {}
    reasoning = f"Router classified as {intent}"

    if intent == "structure_lookup":
        strategy = "structure_lookup"
        scope = "chapter"
        index_hint = "meta"
        reasoning = "User wants to explore concepts; fetching TOC to select a topic."

    elif intent == "book_aggregate":
        strategy = "book_aggregate"
        scope = "book"
        index_hint = "summary"
        summary_sampler = "all"
        reasoning = "User asked for high-level book summary"

    elif intent == "generate_assessment":
        strategy = "generate_assessment"
        scope = "book" 
        index_hint = "summary"
        summary_sampler = "all"
        assessment_params = {"type": "mixed", "count": 10}
        reasoning = "User asked for assessment/quiz"
        
    elif intent == "general_chat":
        strategy = "general_chat"
        scope = "general"
        index_hint = "none" # No retrieval needed
        reasoning = "Conversational or out-of-scope query"

    else: # 'detail'
        strategy = "retrieve"
        index_hint = "detail"
        # Keep legacy keyword check for simple summary requests
        if "summary" in question_lower or "summarize" in question_lower:
            index_hint = "summary"
    
    # Construct the final plan dictionary
    plan = {
        "strategy": strategy,
        "scope": scope,
        "index_hint": index_hint,
        "top_k": 5,
        "top_k_per_chapter": 3,
        "summary_sampler": summary_sampler,
        "expand_to_parents": True,
        "enable_reflection": True,
        "assessment_params": assessment_params,
        "reasoning": reasoning,
        "question": question, # Pass through for retriever
        "book_id": book_id,
        "chapter_id": chapter_id
    }
    # --- END INCREMENT 11.2 LOGIC ---
    
    logger.info(f"[PLANNER] ✅ Plan: strategy={plan['strategy']}, "
                f"scope={plan['scope']}, index={plan['index_hint']}")
    
    # return plan
    return {"plan": plan}

def build_plan(
    question: str,
    question_lower: str,
    scope_analysis: Dict[str, Any],
    intent_analysis: Dict[str, Any],
    book_id: str,
    chapter_id: Optional[str],
    user_preferences: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build execution plan based on scope and intent analysis.
    
    Returns:
        Complete plan dict with strategy, parameters, and reasoning
    """
    scope = scope_analysis["scope"]
    intent_type = intent_analysis["intent_type"]
    
    # Default plan
    plan = {
        "strategy": "retrieve",
        "scope": scope,
        "index_hint": "detail",
        "top_k": 5,
        "top_k_per_chapter": 3,
        "summary_sampler": "chapter",
        "expand_to_parents": True,
        "enable_reflection": True,
        "assessment_params": None,
        "reasoning": ""
    }
    
    # ========== STRATEGY 1: ASSESSMENT GENERATION ==========
    if intent_analysis["is_assessment"]:
        plan.update({
            "strategy": "generate_assessment",
            "index_hint": "summary",  # Use summaries for efficiency
            "summary_sampler": "all",  # Need all chapters
            "expand_to_parents": False,
            "top_k_per_chapter": 1,  # Just summary per chapter
            "assessment_params": extract_assessment_params(question_lower, user_preferences),
            "reasoning": "Assessment generation detected - using summary index across all chapters"
        })
        
        logger.info(f"[PLANNER] Assessment generation: {plan['assessment_params']['num_questions']} questions")
        return plan
    
    # ========== STRATEGY 2: BOOK-LEVEL / MULTI-CHAPTER ==========
    if scope_analysis["requires_aggregation"]:
        plan.update({
            "strategy": "book_aggregate",
            "scope": scope,
            "index_hint": "summary",  # Use summaries for overview
            "summary_sampler": "all",  # Query all chapters
            "expand_to_parents": False,
            "top_k_per_chapter": 2,
            "reasoning": f"{scope.capitalize()}-level query - aggregating across chapters using summary index"
        })
        
        logger.info(f"[PLANNER] Book-level aggregation: summary_sampler=all")
        return plan
    
    # ========== STRATEGY 3: OVERVIEW / SUMMARY ==========
    if intent_analysis["is_overview"]:
        plan.update({
            "strategy": "retrieve",
            "index_hint": "summary",
            "expand_to_parents": False,  # Summaries don't have parents
            "top_k": 3,
            "reasoning": "Overview query - using summary index for high-level context"
        })
        
        logger.info(f"[PLANNER] Overview query: using summary index")
        return plan
    
    # ========== STRATEGY 4: COMPARISON ==========
    if intent_analysis["is_comparison"]:
        plan.update({
            "strategy": "retrieve",
            "index_hint": "detail",  # Need specific details for comparison
            "expand_to_parents": True,
            "top_k": 8,  # More results for comparison
            "enable_reflection": True,
            "reasoning": "Comparison query - using detail index with higher top_k"
        })
        
        logger.info(f"[PLANNER] Comparison query: detail index, top_k=8")
        return plan
    
    # ========== STRATEGY 5: DETAILED EXPLANATION (DEFAULT) ==========
    plan.update({
        "strategy": "retrieve",
        "index_hint": "detail",
        "expand_to_parents": True,  # Get rich parent context
        "top_k": 5,
        "enable_reflection": True,
        "reasoning": "Detailed explanation - using detail index with parent expansion"
    })
    
    logger.info(f"[PLANNER] Default strategy: detail index with parent expansion")
    
    return plan


def extract_assessment_params(
    question_lower: str,
    user_preferences: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract assessment generation parameters from question.
    
    Examples:
    - "Prepare a 20-question paper" → num_questions=20
    - "Create easy questions" → difficulty_filter=easy
    - "HOTS questions only" → hots_only=True
    
    Args:
        question_lower: Lowercased question
        user_preferences: Optional user preferences
    
    Returns:
        Assessment parameters dict
    """
    params = {
        "num_questions": 20,  # Default
        "difficulty_mix": {"easy": 8, "medium": 8, "hard": 4},
        "bloom_mix": {
            "remember": 4,
            "understand": 5,
            "apply": 5,
            "analyze": 3,
            "evaluate": 2,
            "create": 1
        },
        "include_hots": 5,
        "question_types": ["mcq", "short", "long"],
        "marks_distribution": {"1": 8, "2": 6, "3": 4, "5": 2}
    }
    
    # Extract number of questions
    num_match = re.search(r'(\d+)\s*[-]?\s*question', question_lower)
    if num_match:
        params["num_questions"] = int(num_match.group(1))
    
    # Detect difficulty preference
    if "easy" in question_lower and "hard" not in question_lower:
        params["difficulty_mix"] = {"easy": 15, "medium": 5, "hard": 0}
    elif "hard" in question_lower or "difficult" in question_lower:
        params["difficulty_mix"] = {"easy": 2, "medium": 8, "hard": 10}
    elif "medium" in question_lower:
        params["difficulty_mix"] = {"easy": 5, "medium": 12, "hard": 3}
    
    # Detect HOTS preference
    if "hots" in question_lower or "higher order" in question_lower:
        params["include_hots"] = params["num_questions"] // 2  # 50% HOTS
    
    # Detect question type preference
    if "mcq" in question_lower or "multiple choice" in question_lower:
        params["question_types"] = ["mcq"]
    elif "short" in question_lower:
        params["question_types"] = ["short"]
    elif "long" in question_lower or "essay" in question_lower:
        params["question_types"] = ["long"]
    
    # Apply user preferences if available
    if user_preferences:
        if "default_difficulty" in user_preferences:
            diff = user_preferences["default_difficulty"]
            params["difficulty_mix"] = {diff: params["num_questions"], **{k: 0 for k in params["difficulty_mix"] if k != diff}}
        
        if "prefer_hots" in user_preferences and user_preferences["prefer_hots"]:
            params["include_hots"] = params["num_questions"] // 2
    
    logger.debug(f"[PLANNER] Assessment params: {params}")
    
    return params


def should_use_summary_index(question_lower: str) -> bool:
    """
    Determine if summary index should be used.
    
    Returns True for:
    - Overview/summary questions
    - High-level concept questions
    - Topic listing questions
    - Introduction/conclusion questions
    """
    summary_indicators = [
        "overview", "summarize", "summary", "main topics",
        "key concepts", "what topics", "what concepts",
        "introduction", "what does", "what is covered",
        "what are the main", "briefly explain", "in brief"
    ]
    
    return any(kw in question_lower for kw in summary_indicators)


def should_aggregate_chapters(question_lower: str, chapter_id: Optional[str]) -> bool:
    """
    Determine if query requires multi-chapter aggregation.
    
    Returns True for:
    - Explicit "all chapters" mentions
    - Book/subject-level scope
    - No chapter_id specified
    - Cross-chapter comparison
    """
    if chapter_id is None:
        return True
    
    aggregation_indicators = [
        "all chapters", "entire book", "whole book",
        "across chapters", "throughout the book",
        "entire subject", "complete subject",
        "full syllabus", "entire course"
    ]
    
    return any(kw in question_lower for kw in aggregation_indicators)


def estimate_complexity(question: str) -> str:
    """
    Estimate query complexity for parameter tuning.
    
    Returns: "simple", "medium", "complex"
    """
    question_length = len(question)
    num_sentences = question.count('.') + question.count('?') + 1
    
    complex_indicators = [
        "compare", "contrast", "analyze", "evaluate",
        "why", "how", "multiple", "several"
    ]
    
    has_complex_words = any(kw in question.lower() for kw in complex_indicators)
    
    if question_length > 200 or num_sentences > 3 or has_complex_words:
        return "complex"
    elif question_length > 100 or num_sentences > 2:
        return "medium"
    else:
        return "simple"


def adjust_top_k_for_complexity(base_top_k: int, complexity: str) -> int:
    """Adjust top_k based on query complexity"""
    if complexity == "complex":
        return min(base_top_k + 3, 10)
    elif complexity == "simple":
        return max(base_top_k - 2, 3)
    else:
        return base_top_k


def build_plan(
    question: str,
    question_lower: str,
    scope_analysis: Dict[str, Any],
    intent_analysis: Dict[str, Any],
    book_id: str,
    chapter_id: Optional[str],
    user_preferences: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build complete execution plan based on all analysis.
    """
    scope = scope_analysis["scope"]
    intent_type = intent_analysis["intent_type"]
    complexity = estimate_complexity(question)
    
    # Base plan
    plan = {
        "strategy": "retrieve",
        "scope": scope,
        "index_hint": "detail",
        "top_k": 5,
        "top_k_per_chapter": 3,
        "summary_sampler": "chapter",
        "expand_to_parents": True,
        "enable_reflection": True,
        "assessment_params": None,
        "complexity": complexity,
        "reasoning": ""
    }
    
    # ========== ASSESSMENT GENERATION ==========
    if intent_analysis["is_assessment"]:
        assessment_params = extract_assessment_params(question_lower, user_preferences)
        
        plan.update({
            "strategy": "generate_assessment",
            "index_hint": "summary",
            "summary_sampler": "all",
            "expand_to_parents": False,
            "top_k_per_chapter": 1,
            "assessment_params": assessment_params,
            "reasoning": (
                f"Assessment generation: {assessment_params['num_questions']} questions, "
                f"scope={scope}, using summary index for topic coverage"
            )
        })
        
        return plan
    
    # ========== BOOK/SUBJECT-LEVEL AGGREGATION ==========
    if scope_analysis["requires_aggregation"]:
        plan.update({
            "strategy": "book_aggregate",
            "scope": scope,
            "index_hint": "summary",
            "summary_sampler": "all",
            "expand_to_parents": False,
            "top_k_per_chapter": adjust_top_k_for_complexity(2, complexity),
            "reasoning": (
                f"{scope.capitalize()}-level aggregation: querying summary index "
                f"across all chapters with sampler=all"
            )
        })
        
        return plan
    
    # ========== OVERVIEW QUERY ==========
    if intent_analysis["is_overview"]:
        plan.update({
            "strategy": "retrieve",
            "index_hint": "summary",
            "expand_to_parents": False,
            "top_k": adjust_top_k_for_complexity(3, complexity),
            "reasoning": "Overview query: using summary index for high-level context"
        })
        
        return plan
    
    # ========== COMPARISON QUERY ==========
    if intent_analysis["is_comparison"]:
        plan.update({
            "strategy": "retrieve",
            "index_hint": "detail",
            "expand_to_parents": True,
            "top_k": adjust_top_k_for_complexity(8, complexity),
            "enable_reflection": True,
            "reasoning": "Comparison query: detail index with higher top_k and parent expansion"
        })
        
        return plan
    
    # ========== DETAILED EXPLANATION (DEFAULT) ==========
    plan.update({
        "strategy": "retrieve",
        "index_hint": "detail",
        "expand_to_parents": True,
        "top_k": adjust_top_k_for_complexity(5, complexity),
        "enable_reflection": True,
        "reasoning": (
            f"Detailed explanation ({complexity} complexity): "
            f"detail index with parent expansion, top_k={plan['top_k']}"
        )
    })
    
    return plan


def validate_plan(plan: Dict[str, Any]) -> bool:
    """
    Validate plan structure and constraints.
    
    Returns True if plan is valid, False otherwise
    """
    required_fields = ["strategy", "scope", "index_hint", "top_k"]
    
    # Check required fields
    for field in required_fields:
        if field not in plan:
            logger.error(f"[PLANNER] Invalid plan: missing field '{field}'")
            return False
    
    # Validate strategy
    valid_strategies = ["retrieve", "book_aggregate", "generate_assessment"]
    if plan["strategy"] not in valid_strategies:
        logger.error(f"[PLANNER] Invalid strategy: {plan['strategy']}")
        return False
    
    # Validate index_hint
    if plan["index_hint"] not in ["detail", "summary"]:
        logger.error(f"[PLANNER] Invalid index_hint: {plan['index_hint']}")
        return False
    
    # Validate top_k
    if not isinstance(plan["top_k"], int) or plan["top_k"] < 1:
        logger.error(f"[PLANNER] Invalid top_k: {plan['top_k']}")
        return False
    
    return True


# Export public API
__all__ = [
    "plan_step",
    "estimate_complexity",
    "extract_assessment_params",
    "validate_plan"
]
