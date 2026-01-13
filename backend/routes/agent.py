# FILE: backend/routes/agent.py
"""
Agent routes - Corrected for existing ProviderRegistry and Chat History

Existing endpoint:
- POST /answer - LangGraph agent (PRESERVED)

New proactive endpoints:
- POST /greet - Personalized greeting
- GET /suggest-topics - Topic suggestions
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

# Existing imports
from backend.agent.graph_lg import run_agent_graph
from backend.services.telemetry import record_event
from backend.config import get_settings

# Correct import for your registry
from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# ========================================
# UPDATED ENDPOINT TO SUPPORT CHAT HISTORY
# ========================================

class AnswerRequest(BaseModel):
    """Answer request"""
    question: str
    book_id: str
    chapter_id: str
    user_id: str = Field(default="default_user")
    index_hint: Optional[str] = None
    enable_reflection: bool = True
    hots_level: Optional[str] = None
    # ADDED: To support conversational context
    chat_history: Optional[List[Dict[str, str]]] = None
    user_preferences: Optional[Dict[str, Any]] = None 

@router.post("/answer")
async def agent_answer(
    request: AnswerRequest,
    x_mode: Optional[str] = Header(None)
):
    """
    Generate answer using LangGraph agent. Now passes chat history.
    """
    logger.info(f"Answer request: question={request.question[:50]}...")

    record_event(
        "answer_request",
        question=request.question,
        bookid=request.book_id,
        chapterid=request.chapter_id,
        userid=request.user_id,
        history_length=len(request.chat_history) if request.chat_history else 0,
    )

    # UPDATED: Pass chat_history to the agent graph
    response = await run_agent_graph(
        question=request.question,
        book_id=request.book_id,
        chapter_id=request.chapter_id,
        user_id=request.user_id,
        index_hint=request.index_hint,
        enable_reflection=request.enable_reflection,
        hots_level=request.hots_level,
        chat_history=request.chat_history,
        user_preferences=request.user_preferences,
    )

    if x_mode:
        response["meta"]["x_mode"] = x_mode

    return response


# ========================================
# NEW PROACTIVE ENDPOINTS (Preserved)
# ========================================

class GreetRequest(BaseModel):
    user_id: str
    time_of_day: Optional[str] = None

class GreetResponse(BaseModel):
    greeting: str
    suggestions: List[str]
    meta: Dict[str, Any] = {}

@router.post("/greet", response_model=GreetResponse)
async def greet_student(request: GreetRequest):
    """
    Generate personalized greeting using LLM
    """
    try:
        hour = int(request.time_of_day) if request.time_of_day else datetime.now().hour
        
        if hour < 12:
            time_greeting = "Good morning"
            emoji = "🌅"
        elif hour < 17:
            time_greeting = "Good afternoon"
            emoji = "☀️"
        else:
            time_greeting = "Good evening"
            emoji = "🌙"
            
        registry = get_provider_registry()
        
        prompt = f"""You are an enthusiastic NCERT AI Tutor. Generate a warm, friendly greeting for a student.

Guidelines:
- Be encouraging and supportive
- Show enthusiasm about learning
- Keep it concise (2-3 sentences)
- Do NOT use emojis

Time: {time_greeting}
User: {request.user_id}

Generate greeting:"""
        
        try:
            result = registry.generate(prompt=prompt, temperature=0.8, max_tokens=150)
            llm_greeting = result["text"].strip()

            check_text = llm_greeting.lower().strip()
            check_prefix = time_greeting.lower().strip()
            
            if check_text.startswith(check_prefix):
                greeting = f"{emoji} {llm_greeting}"
            else:
                greeting = f"{emoji} {time_greeting}! {llm_greeting}"
                
        except Exception as e:
            logger.warning(f"[GREET] LLM failed: {e}, using fallback")
            greeting = f"{emoji} {time_greeting}! I'm your NCERT AI Tutor, ready to help you learn and explore!"
            
        return GreetResponse(
            greeting=greeting,
            suggestions=[
                "📖 Explain a key concept from the chapter",
                "❓ Test my understanding with questions",
                "🔬 Show me real-world examples",
                "💡 Give me study tips"
            ],
            meta={
                "timestamp": datetime.now().isoformat(),
                "user_id": request.user_id,
                "time_of_day": hour
            }
        )
    
    except Exception as e:
        logger.error(f"[GREET] Error: {e}")
        return GreetResponse(
            greeting="👋 Hello! I'm your NCERT AI Tutor. What would you like to learn today?",
            suggestions=[
                "Explain a concept",
                "Give practice questions",
                "Help with homework",
                "Review a chapter"
            ],
            meta={"fallback": True}
        )

class TopicSuggestResponse(BaseModel):
    topics: List[str]
    source: str

@router.get("/suggest-topics", response_model=TopicSuggestResponse)
async def suggest_topics(book_id: str, chapter_id: str, count: int = 4):
    """
    Generate topic suggestions using LLM
    """
    try:
        shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
        
        summary_text = ""
        for filename in ["chapter_summary.txt", "summary.txt"]:
            summary_file = shard_dir / filename
            if summary_file.exists():
                summary_text = summary_file.read_text(encoding='utf-8')[:1000]
                break
        
        if not summary_text:
            for filename in ["summary_metadata.json", "summary_manifest.json"]:
                json_file = shard_dir / filename
                if json_file.exists():
                    import json
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        summary_text = data.get("summary_text", "")[:1000]
                    if summary_text:
                        break
                        
        registry = get_provider_registry()
        
        if summary_text:
            prompt = f"""Based on this NCERT chapter summary, suggest {count} engaging learning topics or questions.

Chapter Summary:
{summary_text}

Guidelines:
- Make topics engaging and curiosity-driven
- Frame as questions or exploration prompts
- Keep each topic concise (under 8 words)
- Focus on key concepts, not trivial facts

Generate {count} topic suggestions (one per line):"""
        else:
            prompt = f"""Generate {count} engaging NCERT study topics for a Class 10 Science student.

Guidelines:
- Make topics interesting and exploratory
- Keep each under 8 words
- Cover different aspects

Generate {count} topics (one per line):"""
        
        try:
            result = registry.generate(prompt=prompt, temperature=0.7, max_tokens=200)
            
            llm_response = result["text"].strip()
            
            raw_lines = [line.strip() for line in llm_response.split("\n") if line.strip()]
            
            topics = []
            for line in raw_lines:
                lower = line.lower()
                if "here are" in lower or line.endswith(":"):
                    continue
                clean = line.lstrip("0123456789.-) ")
                if clean:
                    topics.append(clean)
            topics = topics[:count]

        except Exception as e:
            logger.warning(f"[SUGGEST] LLM failed: {e}, using fallback")
            topics = []
        
        if not topics:
            topics = [
                "📖 Explain key concepts from the chapter",
                "❓ Test my knowledge with questions",
                "🔬 Show real-world applications",
                "💡 Share effective study strategies"
            ][:count]
        
        return TopicSuggestResponse(
            topics=topics,
            source="llm_generated" if summary_text else "fallback"
        )
    
    except Exception as e:
        logger.error(f"[SUGGEST] Error: {e}")
        return TopicSuggestResponse(
            topics=[
                "Explain concepts from the chapter",
                "Test my knowledge with questions",
                "Show me diagrams and examples",
                "Help me understand difficult topics"
            ][:count],
            source="fallback"
        )

@router.get("/health")
async def agent_health():
    """Health check"""
    return {
        "status": "healthy",
        "endpoints": {
            "existing": ["/agent/answer"],
            "new": ["/agent/greet", "/agent/suggest-topics"]
        },
        "langgraph_enabled": True,
        "timestamp": datetime.now().isoformat()
    }
