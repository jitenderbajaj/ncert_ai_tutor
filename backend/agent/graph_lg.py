# FILE: backend/agent/graph_lg.py

"""
LangGraph-based agent orchestration

Increment 11: planner → retrieve → syllabus_mapper → reflect
→ retrieve_refined → compose → engagement → safety/govern → format
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime
import time

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from backend.config import get_settings
from backend.services.correlation import generate_correlation_id
from backend.services.telemetry import record_event

from backend.agent.steps.planner import plan_step
from backend.agent.steps.retriever import retriever_step
from backend.agent.steps.syllabus import syllabus_mapper_step
from backend.agent.steps.reflect import reflect_step
from backend.agent.steps.compose import compose_step
from backend.agent.steps.engagement import engagement_step
from backend.agent.steps.governance import governance_step
from backend.agent.steps.format import format_step

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentState(TypedDict, total=False):
    """Typed state for agent graph"""

    # Request
    correlation_id: str
    raw_question: str
    effective_question: str 
    question: str
    book_id: str
    chapter_id: str
    user_id: str
    index_hint: Optional[str]
    enable_reflection: bool
    hots_level: Optional[str]
    chat_history: List[Dict[str, str]]
    user_preferences: Optional[Dict[str, Any]]

    # Planner output
    plan: Dict[str, Any]

    # Retriever output
    retrieve_results: List[Dict[str, Any]]
    retrieve_refined_results: Optional[List[Dict[str, Any]]]
    
    # --- ADD THIS LINE ---
    documents: List[Dict[str, Any]]
    # ---------------------

    # Syllabus mapper output
    syllabus_tags: List[str]
    confidence: float

    # Reflect output
    reflection: Optional[Dict[str, Any]]
    needs_refinement: bool

    # Compose output
    draft_answer: str
    citations: List[str]

    # Engagement output
    engagement_meta: Dict[str, Any]

    # Governance output
    governance_verdict: str
    policy_messages: List[str]
    safety_meta: Dict[str, Any]

    # Format output
    final_answer: str
    meta: Dict[str, Any]

    # Agents trace
    agents: List[Dict[str, Any]]



def create_agent_graph() -> StateGraph:
    """Create LangGraph StateGraph for agent orchestration"""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("syllabus_mapper", syllabus_mapper_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("retrieve_refined", retrieve_refined_node)
    workflow.add_node("compose", compose_node)
    workflow.add_node("engagement", engagement_node)
    workflow.add_node("governance", governance_node)
    workflow.add_node("format", format_node)

    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retrieve")
    workflow.add_edge("retrieve", "syllabus_mapper")
    workflow.add_edge("syllabus_mapper", "reflect")

    # Conditional edge after reflect
    workflow.add_conditional_edges(
        "reflect",
        should_refine,
        {
            "refine": "retrieve_refined",
            "compose": "compose",
        },
    )

    workflow.add_edge("retrieve_refined", "compose")
    workflow.add_edge("compose", "engagement")
    workflow.add_edge("engagement", "governance")
    workflow.add_edge("governance", "format")
    workflow.add_edge("format", END)

    return workflow.compile()


async def planner_node(state: AgentState) -> AgentState:
    """Planner node: decide retrieval strategy"""
    logger.info(f"[{state['correlation_id']}] Planner node")
    # start = datetime.utcnow()
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    # FIX: Unwrap the plan dictionary
    step_output = await plan_step(
        question=state["question"],
        book_id=state["book_id"],
        chapter_id=state.get("chapter_id"),
        user_preferences=state.get("user_preferences"),
        chat_history=state.get("chat_history", []),
    )

    # Extract the actual plan dict
    plan = step_output.get("plan", {})
    state["plan"] = plan

    # Preserve raw user input (do not overwrite if already set)
    state["raw_question"] = state.get("raw_question") or state.get("question", "")

    # Compute and store canonical effective question
    effective = plan.get("question") or state["raw_question"]
    state["effective_question"] = effective

    # Promote rewritten/effective question for downstream steps
    state["question"] = effective

    # Compute duration once
    # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    duration_ms = int((time.perf_counter() - start) * 1000)

    state.setdefault("agents", []).append({
        "agent": "planner",
        # "timestamp": start.isoformat(),
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "input": {
            "raw_question": state.get("raw_question"),
            "effective_question": state.get("effective_question"),
            "question": state.get("effective_question"),  # backward compatible
        },
        "output": {
            "strategy": plan.get("strategy"),
            "index_hint": plan.get("index_hint"),
            "scope": plan.get("scope"),
            "top_k": plan.get("top_k"),
            "original_question": plan.get("original_question"),
            "effective_question": plan.get("question"),
            "reasoning": plan.get("reasoning"),
        },
    })

    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="planner",
        durationms=duration_ms,
        indexhint=plan.get("index_hint"),
        strategy=plan.get("strategy"),
        raw_question=state.get("raw_question"),
        original_question=plan.get("original_question"),
        effective_question=state.get("effective_question"),
        reasoning=plan.get("reasoning"),
    )

    return state


async def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve node: fetch relevant passages"""
    logger.info(f"[{state['correlation_id']}] Retrieve node")
    # start = datetime.utcnow()
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    # Use the rewritten question from the PLAN if available
    plan = state.get("plan", {})
    search_query = (
        state.get("effective_question")
        or plan.get("question")
        or state["question"]
    )

    # Create temp state for retriever to use the correct query
    temp_state = state.copy()
    temp_state["question"] = search_query

    step_output = await retriever_step(temp_state)

    # Standardize output to list
    if isinstance(step_output, list):
        results = step_output
    elif isinstance(step_output, dict):
        results = step_output.get("documents") or step_output.get("retrieve_results", [])
    else:
        results = []

    state["retrieve_results"] = results
    state["documents"] = results  # FORCE UPDATE DOCUMENTS

    logger.info(f"DEBUG: retrieve_node set {len(results)} docs in state['documents']")

    # Compute duration once
    # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    duration_ms = int((time.perf_counter() - start) * 1000) 
    

    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="retrieve",
        durationms=duration_ms,
        strategy=state.get("plan", {}).get("strategy"),
        indexhint=state.get("plan", {}).get("index_hint"),
        effective_question=search_query,
        resultscount=len(results),
    )

    state.setdefault("agents", []).append({
        "agent": "retrieve",
        # "timestamp": start.isoformat(),
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "input": {
            "raw_question": state.get("raw_question") or plan.get("original_question"),
            "effective_question": search_query,
            "question": search_query,  # Log actual query used
            "book_id": state.get("book_id"),
            "chapter_id": state.get("chapter_id"),
            "index_hint": state.get("plan", {}).get("index_hint"),
        },
        "output": {
            "results_count": len(results),
            "index_hint": state.get("plan", {}).get("index_hint"),
        },
    })

    return state


async def syllabus_mapper_node(state: AgentState) -> AgentState:
    """Syllabus mapper node: tag with NCERT curriculum"""
    logger.info(f"[{state['correlation_id']}] Syllabus mapper node")
    # start = datetime.utcnow()
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    # await the step!
    mapping = await syllabus_mapper_step(
        question=state["question"],
        retrieve_results=state.get("retrieve_results", []),
        book_id=state["book_id"],
        chapter_id=state["chapter_id"],
        correlation_id=state["correlation_id"],
    )

    # Modify state IN PLACE to preserve 'documents'
    state["syllabus_tags"] = mapping["tags"]
    state["confidence"] = mapping["confidence"]

    # Compute duration once
    # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    duration_ms = int((time.perf_counter() - start) * 1000)

    state.setdefault("agents", []).append({
        "agent": "syllabus_mapper",
        # "timestamp": start.isoformat(),
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "input": {
            "question": state.get("question"),
        },
        "output": {
            "tags": mapping["tags"],
        },
    })

    # Telemetry event for Agent Latency
    logger.info("TELEMETRY_DEBUG syllabus_mapper about to record_event")
    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="syllabus_mapper",
        durationms=duration_ms,
        tagscount=len(mapping.get("tags", [])),
        confidence=mapping.get("confidence"),
    )

    # DEBUG LOG
    logger.info(
        f"DEBUG: syllabus_mapper_node passing through "
        f"{len(state.get('documents', []))} docs"
    )

    return state

async def reflect_node(state: AgentState) -> AgentState:
    """Reflect node: assess answer quality"""
    logger.info(f"[{state['correlation_id']}] Reflect node")
    # start = datetime.utcnow()
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    if not state.get("enable_reflection", True):
        state["needs_refinement"] = False
        state["reflection"] = None
        return state

    # await the step!
    reflection = await reflect_step(
        question=state["question"],
        retrieve_results=state.get("retrieve_results", []),
        syllabus_tags=state.get("syllabus_tags", []),
        confidence=state.get("confidence", 0.0),
        correlation_id=state["correlation_id"],
    )

    state["reflection"] = reflection
    state["needs_refinement"] = reflection["needs_refinement"]

    # Compute duration once
    # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    duration_ms = int((time.perf_counter() - start) * 1000)

    state.setdefault("agents", []).append({
        "agent": "reflect",
        # "timestamp": start.isoformat(),
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "output": {
            "needs_refinement": reflection["needs_refinement"],
        },
    })

    # Telemetry event for Agent Latency
    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="reflect",
        durationms=duration_ms,
        needsrefinement=reflection.get("needs_refinement", False),
    )

    # DEBUG LOG
    logger.info(
        f"DEBUG: reflect_node passing through "
        f"{len(state.get('documents', []))} docs"
    )

    return state


def should_refine(state: AgentState) -> str:
    """Conditional edge: decide if refinement needed"""
    if state.get("needs_refinement", False):
        return "refine"
    return "compose"


async def retrieve_refined_node(state: AgentState) -> AgentState:
    """Retrieve refined node: fetch additional passages based on reflection"""
    logger.info(f"[{state['correlation_id']}] Retrieve refined node")
    # start = datetime.utcnow()
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    reflection = state.get("reflection") or {}
    refined_query = reflection.get("refined_query", state["question"])
    index_hint = state.get("plan", {}).get("index_hint")

    temp_state = state.copy()
    temp_state["question"] = refined_query

    step_output = await retriever_step(temp_state)

    if isinstance(step_output, dict):
        results = step_output.get("documents") or step_output.get("retrieve_results", [])
    else:
        results = step_output if isinstance(step_output, list) else []

    state["retrieve_refined_results"] = results

    # APPEND UNIQUE DOCS TO STATE
    if "documents" not in state:
        state["documents"] = []

    # Create a set of existing IDs to prevent duplicates
    existing_ids = set()
    for doc in state["documents"]:
        if isinstance(doc, dict):
            doc_id = doc.get("metadata", {}).get("chunk_id")
        else:
            doc_id = getattr(doc, "metadata", {}).get("chunk_id", "")
        if doc_id:
            existing_ids.add(doc_id)

    for doc in results:
        if isinstance(doc, dict):
            doc_id = doc.get("metadata", {}).get("chunk_id")
        else:
            doc_id = getattr(doc, "metadata", {}).get("chunk_id", "")

        if not doc_id or doc_id not in existing_ids:
            state["documents"].append(doc)
            if doc_id:
                existing_ids.add(doc_id)

    # Compute duration once
    # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    duration_ms = int((time.perf_counter() - start) * 1000) 

    state.setdefault("agents", []).append({
        "agent": "retrieve_refined",
        # "timestamp": start.isoformat(),
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "input": {
            "raw_question": state.get("raw_question"),
            "effective_question": refined_query,
            "refined_query": refined_query,
            "index_hint": index_hint,
        },
        "output": {
            "results_count": len(results),
        },
    })

    # Telemetry event for Agent Latency
    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="retrieve_refined",
        durationms=duration_ms,
        resultscount=len(results),
        indexhint=index_hint,
    )

    return state

async def compose_node(state: AgentState) -> AgentState:
    """Compose node: generate final answer with LLM"""
    logger.info(f"[{state['correlation_id']}] Compose node")
    # start = datetime.utcnow()
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    # Preserve existing documents if already populated
    documents = state.get("documents", [])

    logger.info(f"DEBUG: compose_node found {len(documents)} docs in state['documents']")
    logger.info(f"DEBUG: retrieve_results in state: {len(state.get('retrieve_results', []))}")

    # Fallback only if documents is empty (legacy path)
    if not documents:
        documents = (
            state.get("retrieve_refined_results")
            or state.get("retrieve_results")
            or []
        )
        state["documents"] = documents

    # Call compose_step via state
    composition = await compose_step(state)

    # Map new output format (final_answer) to legacy state keys (draft_answer)
    state["final_answer"] = composition.get("final_answer")
    state["draft_answer"] = composition.get("final_answer")  # For backward compatibility
    state["citations"] = composition.get("citations", [])

    # Compute duration once
    # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    duration_ms = int((time.perf_counter() - start) * 1000)

    state.setdefault("agents", []).append({
        "agent": "compose",
        # "timestamp": start.isoformat(),
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "input": {
            "question": state.get("effective_question") or state.get("question"),
            "raw_question": state.get("raw_question"),
            "effective_question": state.get("effective_question"),
            "doc_count": len(documents),
        },
        "output": {
            "answer_length": len(state.get("final_answer", "")),
            "provider": composition.get("metadata", {}).get("provider"),
        },
    })

    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="compose",
        durationms=duration_ms,
        citations=len(composition.get("citations", [])),
    )

    return state

# def engagement_node(state: AgentState) -> AgentState:
#     """Engagement node: add beyond-textbook engagement"""
#     logger.info(f"[{state['correlation_id']}] Engagement node")
#     # start = datetime.utcnow()
#     start = time.perf_counter()
#     start_ts = datetime.utcnow().isoformat()

#     engagement = engagement_step(
#         draft_answer=state["draft_answer"],
#         question=state["question"],
#         hots_level=state.get("hots_level"),
#         correlation_id=state["correlation_id"],
#     )

#     state["engagement_meta"] = engagement

#     # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
#     duration_ms = int((time.perf_counter() - start) * 1000)

#     state.setdefault("agents", []).append({
#         "agent": "engagement",
#         # "timestamp": start.isoformat(),
#         "timestamp": start_ts,
#         "duration_ms": duration_ms,
#         "input": {
#             "draft_answer_preview": state.get("draft_answer", "")[:200],
#         },
#         "output": {
#             "interventions": len(engagement.get("interventions", [])),
#         },
#     })

#     record_event(
#         "agent_step",
#         correlationid=state["correlation_id"],
#         agent="engagement",
#         durationms=duration_ms,
#         interventions=len(engagement.get("interventions", [])),
#     )

#     return state
def engagement_node(state: AgentState) -> AgentState:
    """Engagement node: add beyond-textbook engagement"""
    logger.info(f"[{state['correlation_id']}] Engagement node")
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    engagement = engagement_step(
        draft_answer=state["draft_answer"],
        question=state["question"],
        chat_history=state.get("chat_history", []),
        sources=state.get("documents", []),
        hots_level=state.get("hots_level"),
        correlation_id=state["correlation_id"],
    )

    # --- Robust shape guard (prevents KeyError downstream) ---
    if not isinstance(engagement, dict):
        engagement = {
            "interventions": [],
            "boredom_meta": {"detected": False, "score": 0.0, "reasons": ["bad_shape"]},
            "provider_meta": {"provider": "unknown", "model": "unknown"},
        }

    state["engagement_meta"] = engagement

    duration_ms = int((time.perf_counter() - start) * 1000)

    boredom = engagement.get("boredom_meta", {}) or {}
    provider_meta = engagement.get("provider_meta", {}) or {}

    # --- Boredom reasons logging (count + top1) ---
    reasons = boredom.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    boredom_reasons_count = len(reasons)
    boredom_reason_top1 = reasons[0] if reasons else None

    interventions_list = engagement.get("interventions", [])
    if not isinstance(interventions_list, list):
        interventions_list = []
    interventions_count = len(interventions_list)

    state.setdefault("agents", []).append({
        "agent": "engagement",
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "input": {
            "draft_answer_preview": state.get("draft_answer", "")[:200],
            "chat_history_len": len(state.get("chat_history", []) or []),
            "sources_count": len(state.get("documents", []) or []),
            "hots_level": state.get("hots_level"),
        },
        "output": {
            "interventions": interventions_count,
            "boredom_detected": bool(boredom.get("detected", False)),
            "boredom_score": boredom.get("score", 0.0),
            "boredom_reasons_count": boredom_reasons_count,
            "boredom_reason_top1": boredom_reason_top1,
            "provider": provider_meta.get("provider"),
            "model": provider_meta.get("model"),
            "router_reason": provider_meta.get("router_reason"),
            "llm_duration_ms": provider_meta.get("duration_ms", 0),
        },
    })

    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="engagement",
        durationms=duration_ms,
        interventions=interventions_count,
        boredom_detected=bool(boredom.get("detected", False)),
        boredom_score=boredom.get("score", 0.0),
        boredom_reasons_count=boredom_reasons_count,
        boredom_reason_top1=boredom_reason_top1,
        provider=provider_meta.get("provider"),
        model=provider_meta.get("model"),
    )

    return state

def governance_node(state: AgentState) -> AgentState:
    """Governance node: apply safety and coverage checks"""
    logger.info(f"[{state['correlation_id']}] Governance node")
    # start = datetime.utcnow()
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    retrieve_for_govern = (
        state.get("retrieve_refined_results") or state.get("retrieve_results")
    )

    verdict = governance_step(
        draft_answer=state["draft_answer"],
        citations=state["citations"],
        retrieve_results=retrieve_for_govern,
        correlation_id=state["correlation_id"],
    )

    state["governance_verdict"] = verdict["verdict"]
    state["policy_messages"] = verdict["policy_messages"]
    state["safety_meta"] = verdict["safety_meta"]

    # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    duration_ms = int((time.perf_counter() - start) * 1000)

    state.setdefault("agents", []).append({
        "agent": "governance",
        # "timestamp": start.isoformat(),
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "input": {
            "draft_answer_preview": state.get("draft_answer", "")[:200],
            "citations_count": len(state.get("citations", [])),
        },
        "output": {
            "verdict": verdict["verdict"],
            "coverage": verdict["safety_meta"].get("coverage", 0),
        },
    })

    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="governance",
        durationms=duration_ms,
        verdict=verdict["verdict"],
    )

    return state

def format_node(state: AgentState) -> AgentState:
    """Format node: produce final response envelope"""
    logger.info(f"[{state['correlation_id']}] Format node")
    # start = datetime.utcnow()
    start = time.perf_counter()
    start_ts = datetime.utcnow().isoformat()

    formatted = format_step(
        draft_answer=state["draft_answer"],
        citations=state["citations"],
        engagement_meta=state["engagement_meta"],
        governance_verdict=state["governance_verdict"],
        policy_messages=state["policy_messages"],
        safety_meta=state["safety_meta"],
        correlation_id=state["correlation_id"],
    )

    state["final_answer"] = formatted["answer"]
    state["meta"] = formatted["meta"]

    # duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    duration_ms = int((time.perf_counter() - start) * 1000)

    state.setdefault("agents", []).append({
        "agent": "format",
        # "timestamp": start.isoformat(),
        "timestamp": start_ts,
        "duration_ms": duration_ms,
        "input": {
            "draft_answer_preview": state.get("draft_answer", "")[:200],
            "citations_count": len(state.get("citations", [])),
        },
        "output": {
            "final_length": len(formatted["answer"]),
        },
    })

    record_event(
        "agent_step",
        correlationid=state["correlation_id"],
        agent="format",
        durationms=duration_ms,
    )

    return state

# ---------------------------
# Public entrypoint
# ---------------------------

async def run_agent_graph(
    question: str,
    book_id: str,
    chapter_id: str,
    user_id: str,
    index_hint: Optional[str] = None,
    enable_reflection: bool = True,
    hots_level: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    user_preferences: Optional[Dict[str, Any]] = None, 
) -> Dict[str, Any]:
    """Run agent graph and return final response"""

    correlation_id = generate_correlation_id()
    logger.info(f"[{correlation_id}] Starting agent graph for question: {question[:50]}...")
    start_time = datetime.utcnow()

    # Initialize state
    initial_state: AgentState = {
        "correlation_id": correlation_id,
        "question": question,
        "raw_question": question,          # add
        "effective_question": question,    # add (will be updated after rewrite/cleanup)
        "book_id": book_id,
        "chapter_id": chapter_id,
        "user_id": user_id,
        "index_hint": index_hint,
        "enable_reflection": enable_reflection,
        "hots_level": hots_level,
        "chat_history": chat_history or [], 
        "agents": [],
        "user_preferences": user_preferences,
    }

    # Create and run graph
    graph = create_agent_graph()
    final_state = await graph.ainvoke(initial_state)

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    logger.info(f"[{correlation_id}] Agent graph completed in {duration_ms}ms")

  
    # --- PREPARE SOURCES FOR FRONTEND ---
    documents = final_state.get("documents", []) or []
    sources_for_frontend: List[Dict[str, Any]] = []

    for doc in documents:
        if isinstance(doc, dict):
            content = (
                doc.get("page_content")
                or doc.get("pagecontent")
                or doc.get("text")
                or ""
            )
            meta = doc.get("metadata") or {}

            # These exist on expanded parent docs from retrieval_dual.expand_to_parent_documents()
            matchingchildren = (
                doc.get("matchingchildren")
                or doc.get("matching_children")
                or []
            )
            expansion = doc.get("expansion") or {}

            sources_for_frontend.append({
                "content": content,
                "metadata": meta,
                "matchingchildren": matchingchildren,
                "expansion": expansion,
            })
        else:
            content = (
                getattr(doc, "page_content", None)
                or getattr(doc, "pagecontent", None)
                or getattr(doc, "text", None)
                or ""
            )
            meta = getattr(doc, "metadata", {}) or {}

            sources_for_frontend.append({
                "content": content,
                "metadata": meta,
            })
    # -------------------------------------

    # --- DEBUG LOG ---
    logger.info(f"DEBUG: final_state['documents'] count: {len(documents)}")
    logger.info(f"DEBUG: sources_for_frontend count: {len(sources_for_frontend)}")
    # -----------------

    # Build response
    response = {
        "answer": final_state["final_answer"],
        "meta": {
            **final_state["meta"],
            "correlation_id": correlation_id,
            "duration_ms": duration_ms,
            "agents": final_state["agents"],
            "sources": sources_for_frontend, # <--- EXPLICITLY ADDED HERE
        },
    }

    record_event(
        "agent_graph_complete",
        correlationid=correlation_id,
        durationms=duration_ms,
        verdict=final_state.get("governance_verdict"),
    )

    return response

