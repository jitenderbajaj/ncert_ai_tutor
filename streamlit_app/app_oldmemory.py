# FILE: streamlit_app/app.py
"""
NCERT AI Tutor - Enhanced Interactive Experience
Version 0.12.0 - Proactive Tutor + Real-Time Agent Visualization

New Features:
- Proactive LLM-powered tutor personality
- Real-time agent status visualization
- Streaming responses (word-by-word)
- Chat-style interface
- Agent reasoning transparency
- All existing functionality preserved
"""
import streamlit as st
import requests
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import time
import re

import sys
import os
from pathlib import Path

# Add the project root directory to sys.path
# This assumes app.py is in /project_root/streamlit_app/app.py
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

st.session_state.setdefault("debug_ui", False)

from backend.multimodal.tts import get_tts_provider
# from streamlit_app.components.voice_controls import render_voice_controls
# render_voice_controls()

# Page config
st.set_page_config(
    page_title="NCERT AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL
BACKEND_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
.agent-status {
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.agent-working { background-color: #e3f2fd; }
.agent-completed { background-color: #e8f5e9; }
.agent-error { background-color: #ffebee; }
</style>
""", unsafe_allow_html=True)

# st.json(st.session_state.get("last_agent_response"))

if st.session_state.get("debug_ui", False):
    st.json(st.session_state.get("last_agent_response"))

# ========================================
# SESSION STATE INITIALIZATION
# ========================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'messages': [],
        'tutor_greeted': False,
        'agent_status': [],
        'current_agent': None,
        'show_agent_panel': True,
        'conversation_context': [],
        'auto_refresh_io': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Debug UI toggle   
with st.sidebar:
    st.session_state["debug_ui"] = st.toggle(
        "Debug UI",
        value=st.session_state.get("debug_ui", False),
    )

# ========================================
# PROACTIVE TUTOR FUNCTIONS
# ========================================

def get_proactive_greeting() -> Dict[str, Any]:
    """Get a warm, personalized greeting from the tutor"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/agent/greet",
            json={
                "user_id": st.session_state.get("user_id", "student_001"),
                "time_of_day": datetime.now().strftime("%H")
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.sidebar.warning(f"Backend unavailable: {e}")
    
    # Fallback greeting
    hour = datetime.now().hour
    if hour < 12:
        greeting = "🌅 Good morning!"
    elif hour < 17:
        greeting = "☀️ Good afternoon!"
    else:
        greeting = "🌙 Good evening!"
    
    return {
        "greeting": f"{greeting} I'm your NCERT AI Tutor. What would you like to learn today?",
        "suggestions": [
            "Explain a concept from the chapter",
            "Give me practice questions",
            "Help me understand a diagram",
            "Give me study tips"
        ]
    }


def get_topic_suggestions(book_id: str, chapter_id: str) -> List[str]:
    """Get AI-generated topic suggestions based on ingested content"""
    try:
        # Clean version without traces
        url = f"{BACKEND_URL}/agent/suggest-topics"
        params = {"book_id": book_id, "chapter_id": chapter_id}
        
        # st.write(f"🔍 Calling API: {url} with {params}") # <--- REMOVED

        response = requests.get(url, params=params, timeout=10)
        
        # st.write(f"🔍 API Status: {response.status_code}") # <--- REMOVED

        if response.status_code == 200:
            data = response.json()
            topics = data.get("topics", [])
            return topics

    except Exception as e:
        pass
    
    # Fallback suggestions
    return [
        "📖 Explain key concepts",
        "❓ Test my understanding",
        "🔬 Show me examples",
        "💡 Give me study tips"
    ]

def render_engagement_mcqs(meta: Dict[str, Any], msg_key: str):
    import streamlit as st

    engagement_meta = (
        meta.get("engagement_meta")
        or meta.get("engagementmeta")
        or {}
    )

    interventions = engagement_meta.get("interventions") or []
    mcqs = [it for it in interventions if isinstance(it, dict) and it.get("type") == "mcq"]
    if not mcqs:
        return

    st.markdown("#### Practice MCQs")

    for i, it in enumerate(mcqs, start=1):
        mcq = it.get("mcq") or {}
        qid = (mcq.get("id") or f"mcq_{i}").strip()
        question = (mcq.get("question") or "").strip()
        options = mcq.get("options") or []
        # answer_key = (mcq.get("answer_key") or "").strip()
        answer_key = (mcq.get("answer_key") or mcq.get("answerKey") or "").strip()
        explanation = (mcq.get("explanation") or "").strip()

        if not question or not options:
            continue

        base = f"mcq::{msg_key}::{qid}"

        st.markdown(f"**Q{i}.** {question}")

        # Build radio labels; store chosen "A/B/C/D"
        labels = []
        keys = []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            k = (opt.get("key") or "").strip()
            t = (opt.get("text") or "").strip()
            if not k or not t:
                continue
            keys.append(k)
            labels.append(f"{k}) {t}")

        chosen_label = st.radio(
            "Choose one:",
            labels,
            index=None,
            key=f"{base}::choice",
            label_visibility="collapsed",
        )
        chosen_key = chosen_label.split(")")[0].strip() if chosen_label else None

        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            if st.button("Submit", key=f"{base}::submit", width="stretch"):
                st.session_state[f"{base}::submitted"] = True

        with c2:
            # Hide answer until submitted (prevents instant reveal)
            disabled = not st.session_state.get(f"{base}::submitted", False)
            if st.button("Show answer", key=f"{base}::show", disabled=disabled, width="stretch"):
                st.session_state[f"{base}::show_answer"] = True

        submitted = st.session_state.get(f"{base}::submitted", False)
        if submitted:
            if not chosen_key:
                st.warning("Select an option first.")
            elif chosen_key == answer_key:
                st.success("Correct.")
            else:
                st.error("Not quite. Try again or reveal the answer.")

        if st.session_state.get(f"{base}::show_answer", False):
            st.info(f"Answer: {answer_key}")
            if explanation:
                st.caption(explanation)

        st.divider()


def extract_quick_topics(text: str, max_topics: int = 4) -> List[str]:
    """
    Extract up to `max_topics` question-like lines from an LLM block.

    Priority:
    1. If there is a numbered list (1., 2., 3., ...), take those items as topics.
    2. Otherwise, take any lines that look like questions (end with '?').
    3. Never use header/summary lines as topics.
    """
    if not text:
        return []

    topics: List[str] = []

    # ---------- 1) STRICT: numbered questions like "1. ...?" or "2) ...?" ----------
    # Apply per line to avoid spanning across items
    lines = [ln.strip() for ln in text.splitlines()]

    for ln in lines:
        if not ln:
            continue

        # Match "1. question", "2) question", etc.
        m = re.match(r'^\s*(\d+)\s*[\.\)]\s*(.+)', ln)
        if m:
            candidate = m.group(2).strip()
            # Keep as-is; do not over-filter numbered items except basic sanity
            if candidate and candidate not in topics:
                topics.append(candidate)
            continue

    # If we got at least 1 numbered item, return up to max_topics of them
    if topics:
        return topics[:max_topics]

    # ---------- 2) FALLBACK: plain question lines (no numbering) ----------
    topics = []
    for ln in lines:
        if not ln:
            continue

        lower = ln.lower()
        # Skip obvious header/meta lines
        if ("engaging learning topic" in lower or "topic suggestions" in lower
                or "chapter summary" in lower or "generate 4 topic" in lower
                or "based on this ncert" in lower):
            continue

        if ln.endswith("?") and len(ln) >= 10:
            cleaned = re.sub(r'^[\d\.\)\-\*\s•●○■□◆◇]+', '', ln).strip()
            if cleaned and cleaned not in topics:
                topics.append(cleaned)
                if len(topics) >= max_topics:
                    break

    return topics[:max_topics]


# ========================================
# AGENT VISUALIZATION COMPONENTS
# ========================================

def update_agent_status(agent_name: str, state: str, message: str = "", duration_ms: int = 0):
    """Update agent status in session state"""
    status_entry = {
        "agent": agent_name,
        "state": state,
        "message": message,
        "duration_ms": duration_ms,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.agent_status.append(status_entry)
    st.session_state.current_agent = agent_name if state == "working" else None

def render_enhanced_trace(agents_used: List[Dict[str, Any]]):
    """
    Renders a beautiful, grouped view of the agent execution trace.
    Matches the 9-agent workflow structure.
    """
    if not agents_used:
        st.info("No reasoning trace available.")
        return

    # Phase Mappings
    phases = {
        "Strategy": ["planner", "syllabus_mapper"],
        "Discovery": ["retrieve", "reflect", "retrieve_refined"],
        "Synthesis": ["compose"],
        "Assurance": ["engagement", "governance", "format"]
    }

    # Icon Mappings
    icons = {
        "planner": "🗓️", "syllabus_mapper": "📘", "retrieve": "🌊",
        "reflect": "🤔", "retrieve_refined": "🎯", "compose": "📝",
        "engagement": "✨", "governance": "🛡️", "format": "🎨"
    }

    # Group agents by phase
    grouped_trace = {k: [] for k in phases}
    other_agents = []

    for agent in agents_used:
        name = agent.get("agent", "").lower()
        found = False
        for phase, names in phases.items():
            if name in names:
                grouped_trace[phase].append(agent)
                found = True
                break
        if not found:
            other_agents.append(agent)

    # Render Phases
    for phase_name, agents in grouped_trace.items():
        if agents:
            st.caption(f"**{phase_name.upper()}**")
            for ag in agents:
                name = ag.get("agent", "Unknown")
                duration = ag.get("duration_ms", 0)
                icon = icons.get(name.lower(), "⚙️")
                
                # distinct style for long-running tasks (>1s)
                time_style = "color: orange; font-weight: bold;" if duration > 1000 else "color: gray;"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"{icon} **{name.replace('_', ' ').title()}**")
                with col2:
                    st.markdown(f"<span style='{time_style}'>{duration}ms</span>", unsafe_allow_html=True)
            st.markdown("---")
            
    # Render any unknown agents (fallback)
    if other_agents:
        st.caption("**SYSTEM / OTHER**")
        for ag in other_agents:
            name = ag.get("agent", "Unknown")
            duration = ag.get("duration_ms", 0)
            st.write(f"⚙️ {name}: {duration}ms")

def render_detailed_trace(agents_used: List[Dict[str, Any]]):
    """
    Renders a detailed, phase-grouped trace using expanders for each agent,
    displaying Inputs, Outputs, and duration, mimicking the step-by-step handoff.
    """
    import streamlit as st # Ensure Streamlit is imported if this is a separate file
    from typing import Dict, Any, List # Ensure types are imported
    
    if not agents_used:
        st.info("No detailed trace available.")
        return

    # Phase Mappings (Aligned with your 9 LangGraph Agents)
    phases = {
        "🧠 Phase 1: Strategy & Alignment": ["planner", "syllabus_mapper"],
        "🔍 Phase 2: Iterative Discovery": ["retrieve", "reflect", "retrieve_refined"],
        "✍️ Phase 3: Synthesis": ["compose"],
        "🛡️ Phase 4: Assurance": ["engagement", "governance", "format"]
    }

    icons = {
        "planner": "🗓️", "syllabus_mapper": "📘", "retrieve": "🌊",
        "reflect": "🤔", "retrieve_refined": "🎯", "compose": "📝",
        "engagement": "✨", "governance": "🛡️", "format": "🎨"
    }

    # Group agents by phase
    grouped_trace = {k: [] for k in phases}
    for agent in agents_used:
        name = agent.get("agent", "").lower()
        for phase_name, names in phases.items():
            if name in names:
                grouped_trace[phase_name].append(agent)
                break

    # Render Phases and individual agent expanders
    first_agent_rendered = False
    
    for phase_name, agents in grouped_trace.items():
        if agents:
            st.subheader(phase_name)
            
            for agent in agents:
                name = agent.get("agent", "Unknown")
                duration = agent.get("duration_ms", 0)
                icon = icons.get(name.lower(), "⚙️")
                
                # Title for the expander
                title = f"{icon} **{name.replace('_', ' ').title()}** ({duration}ms)"
                
                # Expand the first agent to give an immediate example of detail
                # This mimics the immediate detail of a real-time system
                expanded = True if not first_agent_rendered else False
                
                with st.expander(title, expanded=expanded):
                    col_in, col_out = st.columns(2)
                    
                    # Show Input
                    with col_in:
                        st.markdown("**➡️ Input (State/Context)**")
                        st.json(agent.get("input", {}))
                    
                    # Show Output
                    with col_out:
                        st.markdown("**⬅️ Output (Result)**")
                        st.json(agent.get("output", {}))
                    
                    # Show Agent Reasoning/Internal Thoughts if available
                    if agent.get("reasoning"):
                        st.markdown("**💭 Agent Reasoning/Thoughts**")
                        st.markdown(agent["reasoning"])
                        
                    st.divider()

                first_agent_rendered = True

def render_agent_status_sidebar():
    """Render real-time agent status in sidebar with clickable details"""
    if not st.session_state.show_agent_panel:
        return

    with st.sidebar:
        st.divider()
        st.subheader("🤖 Agent Activity")
        
        # 1. Show Current Working Agent (if any)
        if st.session_state.current_agent:
            st.markdown(f"""
            <div class="agent-status agent-working">
                🔄 <b>{st.session_state.current_agent}</b> working...
            </div>
            """, unsafe_allow_html=True)
        
        # 2. Show Last Completed Agents (Clickable for details)
        # Use the last response's agent trace if available
        last_resp = st.session_state.get("last_agent_response", {})
        agents = last_resp.get("meta", {}).get("agents", []) or last_resp.get("meta", {}).get("agents_used", [])
        
        if agents:
            st.caption("Recent Trace")
            for agent in agents:
                name = agent.get('agent', 'Unknown')
                duration = agent.get('duration_ms', 0)
                
                # Make an expander for each agent in the sidebar
                with st.expander(f"✅ {name} ({duration}ms)"):
                    st.json(agent) # Show full details here
        else:
            st.caption("No recent activity")

def stream_response(text: str, placeholder, step: int = 3):
    displayed = ""
    counter = 0
    for ch in text:
        displayed += ch
        counter += 1
        if counter >= step or ch in {".", "?", "!", "\n"}:
            placeholder.markdown(displayed)
            counter = 0
        time.sleep(0.01)
    placeholder.markdown(displayed)

def pick(d, *keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k, None)
        if v is not None:
            return v
    return default

def handle_new_question(question: str, source: str = None):
    """
    Unified handler with Horizontal Single-Line Agent Visualization.
    Decoupled from button callbacks.
    """
    
    # 1. Prevent duplicate handling
    if 'recent_questions' not in st.session_state:
        st.session_state.recent_questions = set()

    # Check if we are already processing this
    if question in st.session_state.recent_questions:
        return
    st.session_state.recent_questions.add(question)

    # 2. Append User Message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat()
    })

    # 3. Render User Message
    with st.chat_message("user"):
        st.markdown(question)

    # 4. Create Assistant Placeholder
    with st.chat_message("assistant"):
        # Single line container for the Agent Workflow
        workflow_status = st.empty()
        response_placeholder = st.empty()

        try:
            # Define the agent sequence
            agents = [
                "Planner", "Syllabus", "Retrieve", "Reflect", 
                "Refined Search", "Compose", "Review", "Governance", "Format"
            ]
            
            def update_pipeline(active_idx):
                """Helper to render the horizontal pipeline string"""
                steps = []
                for i, agent in enumerate(agents):
                    if i < active_idx:
                        steps.append(f"✅ {agent}")  # Completed
                    elif i == active_idx:
                        steps.append(f"**🔄 {agent}**")  # Active (Bold + Spinner icon)
                    else:
                        steps.append(f"⬜ {agent}")  # Pending
                
                # Join with arrows
                status_text = " → ".join(steps)
                workflow_status.markdown(f"###### {status_text}")

            # === START WORKFLOW SIMULATION ===
            
            # 1. Planner
            update_pipeline(0)
            update_agent_status("Planner", "working", "Decomposing query")
            time.sleep(0.6)
            
            # 2. Syllabus
            update_pipeline(1)
            update_agent_status("Syllabus", "working", "Checking scope")
            time.sleep(0.6)
            
            # 3. Retrieve
            update_pipeline(2)
            update_agent_status("Retrieval", "working", "Initial search")
            time.sleep(0.7)
            
            # 4. Reflect
            update_pipeline(3)
            time.sleep(0.6)
            
            # 5. Refined Search
            update_pipeline(4)
            update_agent_status("Retrieval", "working", "Refined search")
            time.sleep(0.7)
            
            # 6. Compose (The real blocking call)
            update_pipeline(5)
            update_agent_status("Composition", "working", "Generating answer...")
            
            # --- PREPARE CHAT HISTORY (THE FIX) ---
            # Extract standard history from session state messages
            chat_history_payload = []
            # We skip the very last message because that is the current question we just added
            # We take the last 10 messages for context
            previous_messages = st.session_state.messages[:-1] 
            for msg in previous_messages[-10:]:
                role = msg.get("role")
                content = msg.get("content")
                if role in ["user", "assistant"] and content:
                    chat_history_payload.append({"role": role, "content": content})

            # --- BACKEND API CALL ---
            payload = {
                "question": question,
                "book_id": book_id,
                "chapter_id": chapter_id,
                "user_id": user_id,
                "index_hint": index_hint if index_hint != "auto" else None,
                "enable_reflection": enable_reflection,
                "hots_level": hots_level,
                "chat_history": chat_history_payload, # <--- UPDATED KEY & DATA
                "proactive_mode": proactive_mode,
            }
            
            response = requests.post(
                f"{BACKEND_URL}/agent/answer",
                json=payload,
                headers={"X-Mode": "offline"},
                timeout=60,
            )

            if response.status_code == 200:
                # 7. Engagement/Review
                update_pipeline(6)
                update_agent_status("Review", "working", "Polishing tone")
                time.sleep(0.5)
                
                # 8. Governance
                update_pipeline(7)
                time.sleep(0.5)
                
                # 9. Format
                update_pipeline(8)
                update_agent_status("System", "completed", "Workflow finished")
                time.sleep(0.5)
                
                # Final State: All Checked
                steps = [f"✅ {a}" for a in agents]
                workflow_status.markdown(f"###### {' → '.join(steps)}")
                time.sleep(1.0)
                workflow_status.empty() # Clear it to show the answer cleanly
                
            else:
                workflow_status.error(f"❌ Workflow Failed: {response.status_code}")
                return
            
            # === RENDER FINAL ANSWER ===
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]

                st.session_state["last_agent_response"] = data
                st.session_state.conversation_context.append({"question": question, "answer": answer})

                # Robust meta extraction
                meta = pick(data, "meta", default={}) or {}

                # Render answer
                if streaming_enabled:
                    stream_response(answer, response_placeholder)
                else:
                    response_placeholder.markdown(answer)

                # Compute ONCE
                msg_ts = datetime.now().isoformat()
                correlation_id = data.get("correlation_id") or data.get("correlationId")

                # Parse rest of meta
                safety_meta = pick(meta, "safety_meta", "safetymeta", "safetyMeta", default={}) or {}
                coverage = float(pick(safety_meta, "coverage", default=0.0) or 0.0)
                agents_trace = pick(meta, "agents", "agentsUsed", "agentsused", "agents_used", default=[]) or []

                planner = next((a for a in agents_trace if (a.get("agent", "") or "").lower() == "planner"), None)
                planner_out = (planner or {}).get("output") or {}

                strategy = pick(planner_out, "strategy", default=None) or pick(meta, "strategy", default="")
                index_hint_meta = (
                    pick(planner_out, "index_hint", "indexhint", "indexHint", default=None)
                    or pick(meta, "index_hint", "indexhint", "indexHint", default="")
                )

                msg_meta = {
                    "coverage": coverage,
                    "strategy": strategy,
                    "index_hint": index_hint_meta,
                    "indexhint": index_hint_meta,
                    "safety_meta": safety_meta,
                    "safetymeta": safety_meta,
                    "agents_used": agents_trace,
                    "agentsused": agents_trace,
                    "sources": pick(meta, "sources", default=[]) or [],
                    "governance_verdict": pick(meta, "governance_verdict", "governanceverdict", "governanceVerdict", default="") or "",
                    "engagement_meta": pick(meta, "engagement_meta", "engagementmeta", default={}) or {},
                    "correlation_id": correlation_id,
                }

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": msg_ts,
                    "meta": msg_meta,
                })

                # Optional: keep your caption
                if coverage > 0:
                    is_toc = (strategy == "structure_lookup" and index_hint_meta == "meta")
                    label = "Outline-based alignment" if is_toc else "NCERT alignment"
                    st.caption(f"📊 {label}: {coverage:.0%}")

                
        except requests.exceptions.Timeout:
            update_agent_status("Error", "error", "Request timeout")
            st.error("⏱️ Request timed out.")
            
        except Exception as e:
            update_agent_status("Error", "error", str(e))
            st.error(f"❌ Application Error: {e}")

# ========================================
# MAIN SIDEBAR
# ========================================

with st.sidebar:
    st.title("⚙️ Tutor Settings")
    
    # User profile
    st.subheader("👤 Student Profile")
    user_id = st.text_input("User ID", value="student_001", key="user_id_input")
    book_id = st.text_input("Book ID", value="Class10Science", key="book_id_input")
    chapter_id = st.text_input("Chapter ID", value="CH1", key="chapter_id_input")
    
    st.divider()
    
    # Tutor behavior settings
    st.subheader("🎓 Tutor Behavior")
    proactive_mode = st.toggle(
        "Proactive Mode",
        value=True,
        help="Tutor initiates conversations and provides suggestions"
    )
    show_reasoning = st.toggle(
        "Show Agent Reasoning",
        value=True,
        help="Display agent thought process and sources"
    )
    streaming_enabled = st.toggle(
        "Streaming Responses",
        value=True,
        help="Display answers word-by-word like ChatGPT"
    )
    
    st.divider()
    
        # --- VOICE I/O CONTROLS ---
    from streamlit_app.components.voice_controls import render_voice_controls
    render_voice_controls()
    
    st.divider()
    # --------------------------

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        index_hint = st.selectbox(
            "Index Type",
            ["auto", "detail", "summary"],
            index=0,
            help="Which index to use for retrieval"
        )
        enable_reflection = st.checkbox(
            "Enable Reflection",
            value=True,
            help="Agent reflects on answer quality"
        )
        hots_level = st.selectbox(
            "HOTS Level",
            [None, "easy", "medium", "hard"],
            help="Higher-Order Thinking Skills level"
        )

# Render agent status
render_agent_status_sidebar()

# ========================================
# MAIN CONTENT
# ========================================

st.title("🎓 NCERT AI Tutor")
st.caption("Your Interactive Learning Companion | Version 0.12.0 - Enhanced Edition")

# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#     "💬 Chat Tutor",
#     "📥 Ingestion",
#     "🧠 Memory",
#     "🔍 Provider I/O",
#     "🤖 Agent I/O",  
#     "📊 Analytics"
# ])

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💬 Chat Tutor",
    "📥 Ingestion",
    "🧠 Memory",
    "🔍 Provider I/O",
    "🤖 Agent I/O",  
    "📊 Analytics",
    "🕵️ Retrieval Debugger"  # <--- NEW TAB
])

with tab1:
    # ---- 0. Session State Initialization ----
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'greeting_in_progress' not in st.session_state:
        st.session_state.greeting_in_progress = False

    # ---- 1. Helpers ----
    def already_greeted():
        for m in st.session_state.messages:
            if m.get("is_greeting"):
                return True
        return False

    def trace_llm_call(call_type, params=None):
        if st.session_state.get("debug_ui", False):
            print(f"[TRACE] LLM_CALL: {call_type} | {params or ''}")

    # <<<<<< Place this debug print here >>>>>>
    if st.session_state.get("debug_ui", False):
        print(
        f"DEBUG: messages={len(st.session_state.messages)} | "
        f"greeted={already_greeted()} | "
        f"greeting_in_progress={st.session_state.greeting_in_progress}"
    )

    # ---- 2. Proactive Greeting (one-time, rock-solid) ----
    if (
        proactive_mode
        and not already_greeted()
        and not st.session_state.greeting_in_progress
    ):
        st.session_state.greeting_in_progress = True      # Set BEFORE any LLM calls!
        trace_llm_call("proactive_greeting", {"book_id": book_id, "chapter_id": chapter_id})
        greeting_data = get_proactive_greeting()
        greeting_msg = greeting_data.get("greeting", "Hello! How can I help you today?")
        st.session_state.messages.append({
            "role": "assistant",
            "content": greeting_msg,
            "timestamp": datetime.now().isoformat(),
            "suggestions": greeting_data.get("suggestions", []),
            "is_greeting": True,
        })

    # ---- 3. Greeting Banner at Top ----
    greeting_message = None
    for msg in st.session_state.messages:
        if msg.get("is_greeting"):
            greeting_message = msg
            break

    if greeting_message:
        with st.chat_message("assistant"):
            st.markdown(greeting_message["content"])

    # ---- 4. Suggested Topics Bar ----
    suggested = None
    for idx, message in enumerate(st.session_state.messages):
        if message.get("role") == "assistant" and message.get("suggestions"):
            suggested = (idx, message["suggestions"])
            break

    if suggested:
        idx, suggestions = suggested
        st.subheader("💡 Suggested Topics")
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                # Use callback to set pending question
                st.button(
                    suggestion, 
                    key=f"sug_top_{idx}_{i}",
                    on_click=lambda s=suggestion: st.session_state.update(pending_question=s, pending_source="suggested_topic")
                )

    # ======================
    # 5. QUICK TOPICS BAR (With Caching Fix)
    # ======================
    st.divider()
    st.subheader("💡 Quick Topics")
    
    # 1. Initialize topic storage if missing
    if "quick_topics_cache" not in st.session_state:
        st.session_state.quick_topics_cache = []
    if "quick_topics_context" not in st.session_state:
        st.session_state.quick_topics_context = ""

    # 2. Check if we need to fetch new topics (only if empty or chapter changed)
    current_context = f"{book_id}_{chapter_id}"
    
    if not st.session_state.quick_topics_cache or st.session_state.quick_topics_context != current_context:
        try:
            with st.spinner("Generating fresh topics..."):
                raw = get_topic_suggestions(book_id, chapter_id)
                
                # Ensure list format
                if not isinstance(raw, list):
                    raw = [raw]

                # 1. Flatten if it's a single string containing newlines
                if len(raw) == 1 and isinstance(raw[0], str) and "\n" in raw[0]:
                    raw = raw[0].split("\n")
                    
                # 2. Filter and Clean
                new_topics = []
                for item in raw:
                    text = str(item).strip()
                    lower_text = text.lower()
                    
                    # SKIP if it's an intro line
                    if (text.endswith(":") or 
                        "here are" in lower_text or 
                        "topic suggestions" in lower_text):
                        continue
                    
                    # Clean bullets/numbers
                    cleaned = re.sub(r'^[\d\.\)\-\*•●○■□◆◇\s]+', '', text).strip()
                    
                    if cleaned:
                        new_topics.append(cleaned)

                # 3. Clamp to exactly 4 items
                new_topics = new_topics[:4]
                    
                if new_topics:
                    st.session_state.quick_topics_cache = new_topics
                    st.session_state.quick_topics_context = current_context
                    
        except Exception as e:
            st.warning(f"⚠️ Could not load topic suggestions: {e}")


    # 3. Render buttons using the STABLE cache
    topics = st.session_state.quick_topics_cache

    if topics:
        cols = st.columns(len(topics))
        for i, topic in enumerate(topics):
            with cols[i]:
                key = f"quick_topic_{i}_{abs(hash(topic)) % 10000}"
                st.button(
                    topic, 
                    key=key, 
                    width="stretch",
                    on_click=lambda t=topic: st.session_state.update(pending_question=t, pending_source="quick_topic")
                )
    else:
        st.info("💭 Ask me anything about the chapter!")

    st.divider()
    
    # ---- 6. Chat History ----
    chat_container = st.container()

    print("--- All messages in session state at rerun ---")
    for i, m in enumerate(st.session_state.messages):
        print(f"{i}: {m.get('role')} | {(m.get('content') or '')[:40]}")
    print("--- End of messages ---")

    with chat_container:
        for idx, message in enumerate(st.session_state.messages):

            # ✅ FIX: Robust check for greeting
            is_greeting = message.get("is_greeting", False)

            # Safety-net: detect greeting by content if flag missing
            content = message.get("content", "") or ""
            if (not is_greeting) and ("I'm your NCERT AI Tutor" in content) and ("Good morning" in content):
                is_greeting = True

            # SKIP rendering here because it's already shown in the Top Banner
            if is_greeting:
                continue

            print(f"RENDERING MSG({idx}): role={message.get('role')} | {content[:40]}")

            with st.chat_message(message.get("role", "assistant")):
                st.markdown(content)

                if message.get("role") == "assistant":
                    meta = message.get("meta", {}) or {}

                    # Compute once; reuse
                    is_last_message = (idx == len(st.session_state.messages) - 1)
                    msg_key = message.get("timestamp") or f"msg_{idx}"

                    # --- MCQs (interactive) ---
                    # Show only for the last assistant message (recommended)
                    if is_last_message:
                        # st.write("DEBUG engagement_meta keys:", list((meta.get("engagement_meta") or {}).keys()))
                        # st.write("DEBUG interventions:", (meta.get("engagement_meta") or {}).get("interventions"))
                        if st.session_state.get("debug_ui", False):
                            st.write("DEBUG engagement_meta keys:", list((meta.get("engagement_meta") or {}).keys()))
                            st.write("DEBUG interventions:", (meta.get("engagement_meta") or {}).get("interventions"))
                        render_engagement_mcqs(meta, msg_key=msg_key)

                    # --- VOICE I/O: Text-to-Speech ---
                    if st.session_state.get("enable_tts", False) and is_last_message:
                        audio_key = f"tts_{idx}_{len(content)}"
                        try:
                            provider_name = st.session_state.get("tts_provider_select", "pyttsx3")
                            from backend.multimodal.tts import get_tts_provider
                            tts_provider = get_tts_provider(provider_name, fallback="pyttsx3")

                            if content.strip():
                                with st.spinner("🔊"):
                                    tts_result = tts_provider.synthesize(content, correlation_id=audio_key)
                                    if tts_result.get("audio_data"):
                                        st.audio(
                                            tts_result["audio_data"],
                                            format=f"audio/{tts_result.get('format', 'wav')}",
                                        )
                        except Exception as e:
                            st.warning(f"⚠️ Audio error: {e}")

                    # --- Alignment / Coverage UI ---
                    if "coverage" in meta:
                        coverage = float(meta.get("coverage", 0.0) or 0.0)
                        strategy_msg = meta.get("strategy", "") or ""
                        index_hint_msg = (meta.get("index_hint") or meta.get("indexhint") or "")

                        is_toc = (strategy_msg == "structure_lookup" and index_hint_msg == "meta")
                        label = "Outline-based alignment" if is_toc else "NCERT alignment"

                        st.progress(coverage, text=f"📊 {label}: {coverage:.0%}")


    # Handle pending question (unchanged)
    if st.session_state.get("pending_question"):
        q = st.session_state.pending_question
        s = st.session_state.get("pending_source")

        st.session_state.pending_question = None
        st.session_state.pending_source = None

        handle_new_question(q, source=s)
        st.rerun()


    # 2. CHAT INPUT (User Types Here)
    prompt = st.chat_input("Ask me anything about your textbook... 💭")
    if prompt:
        # Handle immediately
        trace_llm_call("chat_input", {"book_id": book_id, "chapter_id": chapter_id, "text": prompt})
        handle_new_question(prompt, source="chat_input")
        st.rerun()


with tab2:
    st.header("📥 PDF Ingestion Wizard")
    st.write("Upload a PDF to ingest with LLM-generated chapter summary")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="ingest_file",
        help="Upload an NCERT chapter PDF"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ingest_book = st.text_input("Book ID", value="Class10Science", key="ingest_book")
    with col2:
        ingest_chapter = st.text_input("Chapter ID", value="CH1", key="ingest_chapter")
    with col3:
        ingest_seed = st.number_input("Seed", value=42, min_value=0, key="ingest_seed")
    
    # Toggle: async vs sync
    use_async = st.toggle(
        "🛰️ Live progress (async)",
        value=True,
        help="Uses /ingest/pdf/async + SSE events. Turn off to use the old synchronous /ingest/pdf call."
    )

    if st.button("🚀 Start Ingestion", type="primary", disabled=uploaded_file is None):
        if uploaded_file:
            progress_bar = st.progress(0, text="Starting ingestion...")
            status_box = st.empty()

            try:
                files = {"file": uploaded_file}
                data = {
                    "book_id": ingest_book,
                    "chapter_id": ingest_chapter,
                    "seed": ingest_seed
                }

                if not use_async:
                    # -------------------- Existing synchronous flow --------------------
                    progress_bar.progress(10, text="Uploading PDF...")
                    progress_bar.progress(20, text="Processing PDF...")

                    response = requests.post(
                        f"{BACKEND_URL}/ingest/pdf",
                        files=files,
                        data=data,
                        timeout=180
                    )

                    progress_bar.progress(90, text="Finalizing...")

                    if response.status_code == 200:
                        progress_bar.progress(100, text="Complete!")
                        result = response.json()
                    else:
                        st.error(f"❌ Ingestion failed: {response.status_code}")
                        with st.expander("Error Details"):
                            st.code(response.text)
                        st.stop()

                else:
                    # -------------------- Async flow: start job + stream SSE --------------------
                    progress_bar.progress(10, text="Submitting async job...")

                    start_resp = requests.post(
                        f"{BACKEND_URL}/ingest/pdf/async",
                        files=files,
                        data=data,
                        timeout=60
                    )

                    if start_resp.status_code != 200:
                        st.error(f"❌ Failed to start async ingestion: {start_resp.status_code}")
                        with st.expander("Error Details"):
                            st.code(start_resp.text)
                        st.stop()

                    start_data = start_resp.json()
                    job_id = start_data.get("job_id")
                    if not job_id:
                        st.error("❌ Async ingestion did not return job_id")
                        st.json(start_data)
                        st.stop()

                    # status_box.info(f"Job started: {job_id}")
                    with status_box.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.success("🚀 **Ingestion job submitted**")
                        with col2:
                            st.caption(f"**ID:** `{job_id[:8]}...`")
                        with col3:
                            st.caption("🛰️ Live progress")


                    # Stream server-sent events (SSE)
                    events_url = f"{BACKEND_URL}/ingest/jobs/{job_id}/events"

                    progress = 15
                    result = None
                    
                    # ✅ Create the debug UI ONCE (not per event)
                    # debug_expander = st.expander("Debug: raw SSE events", expanded=False)
                    # debug_box = debug_expander.empty()

                    # Use a (connect_timeout, read_timeout) tuple so the stream can stay open
                    with requests.get(events_url, stream=True, timeout=(10, 600)) as r:
                        r.raise_for_status()

                        for line in r.iter_lines(decode_unicode=True):
                            if not line:
                                continue

                            # SSE lines look like: "data: {...json...}"
                            if not line.startswith("data:"):
                                continue

                            payload = line[len("data:"):].strip()
                            if not payload:
                                continue

                            evt = json.loads(payload)

                            # Show raw event for debugging (optional)
                            # status_box.write(evt)
                            # with st.expander("Debug: raw SSE events", expanded=False):
                            #     st.write(evt)
                            # ✅ Update the same debug box each event
                            # debug_box.json(evt)

                            # Progress handling: prefer explicit pct if backend emits it
                            if isinstance(evt, dict) and "pct" in evt:
                                try:
                                    progress = max(progress, int(evt["pct"]))
                                except Exception:
                                    pass
                            else:
                                progress = min(95, progress + 5)

                            msg = evt.get("message") or evt.get("step") or evt.get("type") or "Working..."
                            progress_bar.progress(min(progress, 99), text=str(msg))

                            etype = evt.get("type")
                            if etype in ("ingest_complete", "ingestjob_complete"):
                                result = evt.get("result") or {}
                                # ✅ normalize to the same shape as sync route
                                if "detail_count" not in result:
                                    detail_manifest = result.get("detail_manifest", {}) or {}
                                    summary_manifest = result.get("summary_manifest", {}) or {}
                                    images = result.get("images", []) or []
                                    result["detail_count"] = (detail_manifest.get("stats") or {}).get("num_chunks", 0)
                                    result["summary_count"] = (summary_manifest.get("stats") or {}).get("num_chunks", 0)
                                    result["image_count"] = len(images)
                                progress_bar.progress(100, text="Complete!")
                                break


                            if etype == "ingest_error":
                                st.error(f"❌ Ingestion failed: {evt.get('error', 'Unknown error')}")
                                if st.session_state.get("debug_ui", False):
                                    st.json(evt)
                                    st.stop()

                    if result is None:
                        st.error("❌ SSE stream ended without ingest_complete.")
                        st.stop()

                # -------------------- Render results (unchanged) --------------------
                st.success("✅ Ingestion successful!")
                
                detail_count = result.get("detail_count", 0)
                summary_count = result.get("summary_count", 0)
                image_count = result.get("image_count", 0)
                summary_text = result.get("summary_text", "")
                
                pcs = result.get("parent_child_stats") or {}
                num_parents = pcs.get("num_parents", 0)
                num_children = pcs.get("num_children", 0)

                # col1, col2, col3, col4 = st.columns(4)
                # with col1:
                #     st.metric("Detail Chunks", detail_count)
                # with col2:
                #     st.metric("Summary Chunks", summary_count)
                # with col3:
                #     st.metric("Images", image_count)
                # with col4:
                #     st.metric("Summary Length", f"{len(summary_text):,} chars")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1: st.metric("🧩 Parents", num_parents)
                with col2: st.metric("📄 Children", num_children) 
                with col3: st.metric("Detail", detail_count)
                with col4: st.metric("Summary", summary_count)
                with col5: st.metric("🖼️ Images", image_count)
                with col6: st.metric("📝 Summary", f"{len(summary_text):,} chars")

                toc = result.get("toc")
                if toc:
                    st.subheader("📑 Table of Contents")
                    with st.expander("View TOC", expanded=False):
                        st.json(toc)
                
                if summary_text:
                    st.subheader("📝 LLM-Generated Summary")
                    with st.expander("View Chapter Summary", expanded=True):
                        st.text_area(
                            "Summary Content",
                            value=summary_text,
                            height=250,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                else:
                    st.warning("No summary text received from backend")

                detail_manifest = result.get("detail_manifest", {})
                summary_manifest = result.get("summary_manifest", {})

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📄 Detail Index")
                    with st.expander("View Manifest"):
                        st.json(detail_manifest)

                with col2:
                    st.subheader("📋 Summary Index")
                    with st.expander("View Manifest"):
                        st.json(summary_manifest)

                images = result.get("images", [])
                if images:
                    st.subheader(f"🖼️ Extracted Images ({len(images)})")
                    with st.expander("View Images"):
                        for i, img in enumerate(images[:10]):
                            st.json(img)
                        if len(images) > 10:
                            st.info(f"... and {len(images) - 10} more")

            except requests.exceptions.Timeout:
                st.error("⏱️ Ingestion timed out. The file may be too large.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Please upload a PDF file first")


# ========================================
# TAB 3: MEMORY (Preserved from original)
# ========================================

with tab3:
    st.header("🧠 Memory Management")
    st.write("Store and retrieve user-specific learning context")
    
    col1, col2 = st.columns(2)
    
    # Store memory
    with col1:
        st.subheader("💾 Store Memory")
        mem_user = st.text_input("User ID", value="student_001", key="mem_store_user")
        mem_chapter = st.text_input("Chapter ID (optional)", value="", key="mem_store_chapter")
        mem_key = st.text_input("Key", value="learning_style", key="mem_key")
        mem_value = st.text_input("Value", value="visual learner", key="mem_value")
        mem_ttl = st.number_input("Retention (days)", value=90, min_value=1, key="mem_ttl")
        
        if st.button("💾 Store", key="mem_store_btn", width="stretch"):
        # if st.button("💾 Store", key="mem_store_btn", width="stretch"):
            try:
                payload = {
                    "user_id": mem_user,
                    "chapter_id": mem_chapter if mem_chapter else None,
                    "key": mem_key,
                    "value": mem_value,
                    "retention_ttl_days": mem_ttl
                }
                
                response = requests.post(
                    f"{BACKEND_URL}/memory/put",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    st.success("✅ Memory stored successfully!")
                else:
                    st.error(f"❌ Failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Retrieve memory
    with col2:
        st.subheader("🔍 Retrieve Memory")
        get_user = st.text_input("User ID", value="student_001", key="mem_get_user")
        get_chapter = st.text_input("Chapter ID (optional)", value="", key="mem_get_chapter")
        get_key = st.text_input("Key", value="learning_style", key="mem_get_key")
        
        if st.button("🔍 Retrieve", key="mem_get_btn", width="stretch"):
        # if st.button("🔍 Retrieve", key="mem_get_btn", width="stretch"):
            try:
                params = {"user_id": get_user, "key": get_key}
                if get_chapter:
                    params["chapter_id"] = get_chapter
                
                response = requests.get(
                    f"{BACKEND_URL}/memory/get",
                    params=params,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        st.success(f"✅ Found: `{result['value']}`")
                        if "stored_at" in result:
                            st.caption(f"Stored at: {result['stored_at']}")
                    else:
                        st.warning("⚠️ Memory not found")
                else:
                    st.error(f"❌ Failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

with tab4:
    st.header("🔄 Provider I/O Inspector")
    st.write("View LLM prompts and responses for debugging")
    
    # Controls row
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        limit = st.number_input("Limit", value=20, min_value=1, max_value=100, key="io_limit")
    with col2:
        if st.button("🔄 Refresh", key="io_refresh"):
            st.rerun()
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="io_auto")
    
    try:
        # Fetch provider I/O entries from backend
        response = requests.get(
            f"{BACKEND_URL}/provider-io/recent",
            params={"limit": limit},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Handle both response formats
            if isinstance(data, dict):
                entries = data.get("entries", [])
            else:
                entries = data
            
            # Import and use the proper component
            from components.provider_io import render_provider_io_panel
            
            # Render using the component (with all unique keys)
            render_provider_io_panel(entries)
            
        else:
            st.error(f"❌ Failed to fetch: {response.status_code}")
    
    except Exception as e:
        st.error(f"❌ Error loading Provider I/O data: {e}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(2)
        st.rerun()


# =======================================
# TAB 5: AGENT I/O INSPECTOR
# =======================================

with tab5:
    st.header("🔬 Agent I/O Inspector")
    st.markdown("View the complete, step-by-step transition of state between the 9 LangGraph agents.")

    # Get the last agent response data from session state
    last_response = st.session_state.get("last_agent_response", {})
    # agents = last_response.get("meta", {}).get("agents_used", [])
    # Try both keys just in case
    meta = last_response.get("meta", {})
    agents = meta.get("agents", []) or meta.get("agents_used", [])


    if not agents:
        st.info("No agent trace available. Ask a question in the **Chat** tab first to generate a trace.")
    else:
        st.info("The trace below shows the full data (Input/Output JSON) passed between each agent node.")
        
        # --- Define Agent Descriptions/Icons (needed for a nice visual) ---
        agent_icons = {
            "planner": "🗓️", "syllabus_mapper": "📘", "retrieve": "🌊",
            "reflect": "🤔", "retrieve_refined": "🎯", "compose": "📝",
            "engagement": "✨", "governance": "🛡️", "format": "🎨"
        }
        # --- End Definitions ---

        for idx, agent in enumerate(agents):
            name = agent.get('agent', 'Unknown')
            duration = agent.get('duration_ms', 0)
            icon = agent_icons.get(name.lower(), "⚙️")
            
            title = f"{icon} **{name.replace('_', ' ').title()}** ({duration} ms)"
            
            # Expand the first agent to immediately show detail
            with st.expander(title, expanded=True if idx == 0 else False):
                
                # 1. Internal Reasoning/Thoughts (If available)
                if agent.get("reasoning"):
                    st.caption("💭 **Agent Thoughts/Reasoning**")
                    st.markdown(agent["reasoning"])
                    st.divider()

                # 2. Input/Output Columns for State Transition
                col_in, col_out = st.columns(2)

                with col_in:
                    st.markdown("**➡️ Inputs (Received State)**")
                    # Use st.json for easy viewing of large data
                    st.json(agent.get("input", {}))

                with col_out:
                    st.markdown("**⬅️ Outputs (State Delta)**")
                    st.json(agent.get("output", {}))

        # Optional: show raw state for debugging
        with st.expander("Raw Agent Trace JSON (Debug)", expanded=False):
            st.json(agents)

# ========================================
# TAB 6: ANALYTICS
# ========================================

with tab6:
    st.header("📊 Analytics Dashboard")

    # ---- Controls ----
    colc1, colc2, colc3, colc4 = st.columns(4)
    with colc1:
        since_hours = st.selectbox(
            "Window",
            options=[1, 6, 12, 24, 72],
            index=3,
            format_func=lambda h: f"{h}h",
            key="metrics_since_hours",
        )
    with colc2:
        filt_bookid = st.text_input("Book ID filter", value="", key="metrics_bookid")
    with colc3:
        filt_chapterid = st.text_input("Chapter ID filter", value="", key="metrics_chapterid")
    with colc4:
        filt_userid = st.text_input("User ID filter", value="", key="metrics_userid")

    params = {"since_hours": since_hours}
    if filt_bookid:
        params["bookid"] = filt_bookid
    if filt_chapterid:
        params["chapterid"] = filt_chapterid
    if filt_userid:
        params["userid"] = filt_userid

    # ---- Fetch metrics from backend ----
    summary = None
    agents_data = None
    runs_data = None

    try:
        with st.spinner("Loading global metrics..."):
            # Summary
            resp_sum = requests.get(f"{BACKEND_URL}/metrics/summary", params=params, timeout=10)
            if resp_sum.status_code == 200:
                summary = resp_sum.json()

            # Agents
            resp_agents = requests.get(f"{BACKEND_URL}/metrics/agents", params=params, timeout=10)
            if resp_agents.status_code == 200:
                agents_data = resp_agents.json()

            # Runs (fixed limit for dashboard)
            runs_params = dict(params)
            runs_params["limit"] = 50
            resp_runs = requests.get(f"{BACKEND_URL}/metrics/runs", params=runs_params, timeout=10)
            if resp_runs.status_code == 200:
                runs_data = resp_runs.json()

    except Exception as e:
        st.error(f"Failed to load metrics: {e}")

    if not summary:
        st.info("No telemetry data available yet for this window.")
        st.stop()

    runs = summary.get("runs", {})
    retrieval = summary.get("retrieval", {})
    reflection = summary.get("reflection", {})
    governance = summary.get("governance", {})

    # ---- KPI Row ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Runs (completed)", runs.get("count", 0))
    with col2:
        st.metric("p95 Latency", f"{runs.get('p95_duration_ms', 0.0):.0f} ms")
    with col3:
        zero_rate = retrieval.get("zero_result_rate", 0.0) * 100
        st.metric("Zero-Result Searches", f"{zero_rate:.1f}%")
    with col4:
        refl_rate = reflection.get("rate", 0.0) * 100
        st.metric("Reflection Trigger Rate", f"{refl_rate:.1f}%")

    st.markdown("---")

    # ---- Agent Latency Table ----
    st.subheader("⏱ Agent Latency (global)")
    if agents_data and agents_data.get("agents"):
        rows = agents_data["agents"]
        # Convert to a simple table
        agent_rows = [
            {
                "Agent": r["agent"],
                "Count": r["count"],
                "Avg (ms)": f"{r['avg_duration_ms']:.0f}",
                "p95 (ms)": f"{r['p95_duration_ms']:.0f}",
                "Max (ms)": f"{r['max_duration_ms']:.0f}",
            }
            for r in rows
        ]
        st.table(agent_rows)
    else:
        st.info("No agent_step events found in this window.")

    st.markdown("---")

    # ---- Strategy & Governance ----
    colsx, colsy = st.columns(2)
    with colsx:
        st.subheader("📚 Routing Strategies")
        strat = summary.get("routing", {}).get("strategy_counts", {})
        if strat:
            strat_rows = [{"Strategy": k, "Count": v} for k, v in strat.items()]
            st.table(strat_rows)
        else:
            st.write("No strategy data.")

    with colsy:
        st.subheader("🛡 Governance Verdicts")
        verd = governance.get("verdict_counts", {})
        if verd:
            verd_rows = [{"Verdict": k, "Count": v} for k, v in verd.items()]
            st.table(verd_rows)
        else:
            st.write("No governance verdicts recorded.")

    st.markdown("---")

    # ---- Recent Runs ----
    st.subheader("🧪 Recent Runs")
    if runs_data and runs_data.get("runs"):
        runs_rows = []
        for r in runs_data["runs"]:
            runs_rows.append(
                {
                    "Time": r.get("ts", ""),
                    "Correlation ID": r.get("correlationid", ""),
                    "User": r.get("userid", ""),
                    "Book": r.get("bookid", ""),
                    "Chapter": r.get("chapterid", ""),
                    "Duration (ms)": r.get("durationms", 0),
                    "Strategy": r.get("strategy") or "",
                    "Index": r.get("indexhint") or "",
                    "Verdict": r.get("verdict") or "",
                }
            )
        st.dataframe(runs_rows, width="stretch")
        # st.dataframe(runs_rows, width="stretch")
    else:
        st.info("No recent runs found for this window.")

# ========================================
# TAB 7: RETRIEVAL DEBUGGER
# ========================================
# with tab7:
#     st.header("🕵️ Retrieval Deep Dive")
#     st.caption("Inspect exactly what was searched and what chunks were fed to the LLM.")

#     last_response = (
#         st.session_state.get("last_agent_response")
#         or st.session_state.get("lastagentresponse")
#         or {}
#     )

#     if not last_response:
#         st.info("No query has been run yet. Ask a question in the Chat Tutor to populate this tab.")
#     else:
#         meta = last_response.get("meta", {}) or {}
#         agents_trace = meta.get("agents") or meta.get("agents_used") or meta.get("agentsUsed") or []
#         agents_trace = agents_trace if isinstance(agents_trace, list) else []
#         # ---- Tab7 UI toggles ----
#         show_all_children = st.toggle(
#             "Show ALL child chunks under each parent (debug)",
#             value=False,
#             help="Off by default. Turn on only when debugging full parent→all-children expansion."
#         )
#         # ---- End UI toggles ----
#         def g(d: dict, *keys, default=None):
#             if not isinstance(d, dict):
#                 return default
#             for k in keys:
#                 if k in d and d.get(k) is not None:
#                     return d.get(k)
#             return default

#         # 1. FIND RETRIEVAL STEPS (support naming variants)
#         retrieval_agent_names = {
#             "retrieve",
#             "retrieve_refined",
#             "retrieverefined",
#             "structure_lookup",
#             "structurelookup",
#         }
#         retrieval_steps = [
#             a for a in agents_trace
#             if (a.get("agent") or "").lower() in retrieval_agent_names
#         ]

#         if not retrieval_steps:
#             st.warning("No retrieval steps found in the last execution.")

#         # 2. ITERATE THROUGH SEARCHES (Initial vs Refined)
#         for i, step in enumerate(retrieval_steps):
#             agent_name = (step.get("agent", "Unknown") or "Unknown").replace("_", " ").title()
#             duration = step.get("duration_ms", 0)

#             with st.expander(f"🔍 Search #{i+1}: {agent_name} ({duration}ms)", expanded=True):
#                 input_data = step.get("input", {}) or {}

#                 raw_q = g(input_data, "raw_question", "rawquestion", default="N/A")
#                 effective_q = g(
#                     input_data,
#                     "effective_question", "effectivequestion",
#                     "question",
#                     "refined_query", "refinedquery",
#                     default="N/A",
#                 )
#                 index_hint = g(input_data, "index_hint", "indexhint", default="auto")

#                 st.markdown("### 1. Vector Search Parameters")
#                 col1, col2, col3 = st.columns([3, 3, 1])

#                 with col1:
#                     st.text_input(f"Raw User Input (#{i+1})", value=raw_q, disabled=True)

#                 with col2:
#                     st.text_input(f"Query Used (Effective) (#{i+1})", value=effective_q, disabled=True)

#                 with col3:
#                     st.text_input(f"Index Hint (#{i+1})", value=index_hint, disabled=True)

#                 output_data = step.get("output", {}) or {}
#                 result_count = g(output_data, "results_count", "resultscount", "resultsCount", default=0)

#                 st.markdown(f"### 2. Retrieved Chunks (this step: {result_count})")

#                 # LAST retrieval step: show final context chunks passed to LLM + ONLY initial vector-hit children per parent
#                 if i == len(retrieval_steps) - 1:
#                     st.info("Showing chunks passed to LLM (final context window).")

#                     # In your backend response shape, sources are stored under meta["sources"]
#                     sources = (
#                         meta.get("sources")
#                         or last_response.get("sources")     # fallback if you ever move it top-level
#                         or meta.get("documents")
#                         or last_response.get("documents")
#                         or []
#                     )

#                     with st.expander("DEBUG: sources[0] keys", expanded=False):
#                         if sources and isinstance(sources[0], dict):
#                             st.write(sorted(list(sources[0].keys())))
#                             st.json(sources[0])

#                     if not sources:
#                         st.write("No sources returned in metadata/response.")
#                     else:
#                         for idx, source in enumerate(sources):
#                             # Normalize
#                             if isinstance(source, str):
#                                 content = source
#                                 meta_info = {}
#                             else:
#                                 content = source.get("content") or source.get("text") or "No Content"
#                                 meta_info = source.get("metadata", {}) or {}

#                             # Try hard to find useful labels even if backend only sends content+metadata
#                             chunktype = (
#                                 g(source, "chunktype", "chunk_type", default=None)
#                                 or g(meta_info, "chunktype", "chunk_type", default="unknown")
#                             )
#                             score = (
#                                 g(source, "score", default=None)
#                                 or g(meta_info, "score", default="N/A")
#                             )
#                             parent_id = (
#                                 g(source, "id", "passage_id", "passageid", default=None)
#                                 or g(meta_info, "id", "passage_id", "passageid", default=None)
#                                 or "unknown"
#                             )

#                             chunk_label = f"Chunk {idx+1} — type={chunktype}, parent_id={parent_id}, score={score}"

#                             st.markdown(f"**{chunk_label}**")
#                             st.text_area(
#                                 label=f"Content {idx+1}",
#                                 value=content,
#                                 height=150,
#                                 key=f"chunk_{i}_{idx}",
#                                 disabled=True,
#                             )

#                             with st.popover(f"Metadata {idx+1}"):
#                                 st.json(source if isinstance(source, dict) else meta_info)

#                             # CHILD HITS: only the child chunks that were retrieved by vector search and triggered this parent
#                             if isinstance(source, dict):
#                                 matching = (
#                                     source.get("matchingchildren")
#                                     or source.get("matching_children")
#                                     or meta_info.get("matchingchildren")
#                                     or meta_info.get("matching_children")
#                                     or []
#                                 )

#                                 exp = source.get("expansion") or meta_info.get("expansion") or {}
#                                 from_children = exp.get("fromchildren") or exp.get("from_children") or []

#                                 if matching or from_children:
#                                     count = len(matching) if matching else len(from_children)
#                                     with st.expander(
#                                         f"Child chunks retrieved initially (triggered this parent) ({count})",
#                                         expanded=False,
#                                     ):
#                                         if matching:
#                                             for j, c in enumerate(matching):
#                                                 child_id = g(c, "childid", "child_id", default="unknown")
#                                                 child_score = g(c, "score", default="N/A")
#                                                 child_text = g(c, "text", default="") or g(c, "textpreview", "text_preview", default="")
#                                                 preview = g(c, "textpreview", "text_preview", default="")

#                                                 st.markdown(f"- {child_id} | score={child_score}")

#                                                 # Optional: keep preview as a one-liner
#                                                 if preview and not child_text:
#                                                     st.caption(preview)


#                                                 # Full matched child text (this is what you want)
#                                                 if child_text:
#                                                     st.text_area(
#                                                         label=f"Matched child text {idx+1}.{j+1}",
#                                                         value=child_text,
#                                                         height=220,
#                                                         key=f"child_full_{i}_{idx}_{j}",
#                                                         disabled=True,
#                                                     )
#                                         else:
#                                             # fallback: show IDs only if previews weren't carried through
#                                             for child_id in from_children:
#                                                 st.write(f"- {child_id}")
                                                                
#                                 # FULL CHILD CHUNKS (debug only): all children under this parent (full text)
#                                 if show_all_children:
#                                     child_chunks = (
#                                         source.get("child_chunks")
#                                         or source.get("childchunks")
#                                         or meta_info.get("child_chunks")
#                                         or meta_info.get("childchunks")
#                                         or []
#                                     )

#                                     if child_chunks:
#                                         with st.expander(f"All child chunks under this parent ({len(child_chunks)})", expanded=False):
#                                             for j, ch in enumerate(child_chunks):
#                                                 if not isinstance(ch, dict):
#                                                     continue

#                                                 ch_id = g(ch, "id", "chunk_id", "chunkid", default=f"child_{j+1}")
#                                                 ch_text = g(ch, "text", "page_content", "content", default="")
#                                                 ch_meta = ch.get("metadata", {}) or {}

#                                                 st.markdown(f"- {ch_id}")
#                                                 if ch_text:
#                                                     st.text_area(
#                                                         label=f"Child text {idx+1}.{j+1}",
#                                                         value=ch_text,
#                                                         height=140,
#                                                         key=f"child_full_all_{i}_{idx}_{j}",  # key tweak to avoid collision
#                                                         disabled=True,
#                                                     )
#                                                 with st.popover(f"Child metadata {idx+1}.{j+1}"):
#                                                     st.json(ch_meta)

#                             st.divider()
#                 else:
#                     st.caption(
#                         "Intermediate retrieval results from this step are merged "
#                         "into the final context shown for the last search above."
#                     )

#         # 3. LLM CONTEXT PREVIEW
#         st.header("🧠 Context Passed to LLM")
#         compose_step = next((a for a in agents_trace if a.get("agent") == "compose"), None)

#         if compose_step:
#             st.success("Found composition step. Showing full input state for the LLM.")
#             st.json(compose_step.get("input", {}) or {})
#         else:
#             st.warning("No composition step found in the trace.")

# ========================================
# TAB 7: RETRIEVAL DEBUGGER (NO NESTED EXPANDERS)
# ========================================
with tab7:
    st.header("🕵️ Retrieval Deep Dive")
    st.caption("Inspect exactly what was searched and what chunks were fed to the LLM.")

    last_response = (
        st.session_state.get("last_agent_response")
        or st.session_state.get("lastagentresponse")
        or {}
    )

    if not last_response:
        st.info("No query has been run yet. Ask a question in the Chat Tutor to populate this tab.")
    else:
        meta = last_response.get("meta", {}) or {}

        agents_trace = (
            meta.get("agents")
            or meta.get("agents_used")
            or meta.get("agentsUsed")
            or []
        )
        agents_trace = agents_trace if isinstance(agents_trace, list) else []

        show_all_children = st.toggle(
            "Show ALL child chunks under each parent (debug)",
            value=False,
            help="Off by default. Turn on only when debugging full parent→all-children expansion."
        )

        def g(d: dict, *keys, default=None):
            if not isinstance(d, dict):
                return default
            for k in keys:
                if k in d and d.get(k) is not None:
                    return d.get(k)
            return default

        # 1) FIND RETRIEVAL STEPS (support naming variants)
        retrieval_agent_names = {
            "retrieve",
            "retrieve_refined",
            "retrieverefined",
            "structure_lookup",
            "structurelookup",
        }
        retrieval_steps = [
            a for a in agents_trace
            if (a.get("agent") or "").lower() in retrieval_agent_names
        ]

        if not retrieval_steps:
            st.warning("No retrieval steps found in the last execution.")

        # 2) ITERATE THROUGH SEARCHES (Initial vs Refined)
        for i, step in enumerate(retrieval_steps):
            agent_name = (step.get("agent", "Unknown") or "Unknown").replace("_", " ").title()
            duration = step.get("duration_ms", 0)

            # OUTER EXPANDER ONLY (no nested expanders inside)
            with st.expander(f"🔍 Search #{i+1}: {agent_name} ({duration}ms)", expanded=True):
                input_data = step.get("input", {}) or {}

                raw_q = g(input_data, "raw_question", "rawquestion", default="N/A")
                effective_q = g(
                    input_data,
                    "effective_question", "effectivequestion",
                    "question",
                    "refined_query", "refinedquery",
                    default="N/A",
                )
                index_hint = g(input_data, "index_hint", "indexhint", default="auto")

                st.markdown("### 1. Vector Search Parameters")
                c1, c2, c3 = st.columns([3, 3, 1])
                with c1:
                    st.text_input(f"Raw User Input (#{i+1})", value=raw_q, disabled=True, key=f"tab7_raw_{i}")
                with c2:
                    st.text_input(f"Query Used (Effective) (#{i+1})", value=effective_q, disabled=True, key=f"tab7_eff_{i}")
                with c3:
                    st.text_input(f"Index Hint (#{i+1})", value=index_hint, disabled=True, key=f"tab7_hint_{i}")

                output_data = step.get("output", {}) or {}
                result_count = g(output_data, "results_count", "resultscount", "resultsCount", default=0)

                st.markdown(f"### 2. Retrieved Chunks (this step: {result_count})")

                # Only the LAST retrieval step shows the final context window
                if i == len(retrieval_steps) - 1:
                    st.info("Showing chunks passed to LLM (final context window).")

                    sources = (
                        meta.get("sources")
                        or last_response.get("sources")
                        or meta.get("documents")
                        or last_response.get("documents")
                        or []
                    )

                    # Debug sources WITHOUT an expander (use popover)
                    with st.popover("Debug: sources[0]"):
                        if sources and isinstance(sources[0], dict):
                            st.write(sorted(list(sources[0].keys())))
                            st.json(sources[0])
                        else:
                            st.write("No dict-like sources[0] to inspect.")

                    if not sources:
                        st.write("No sources returned in metadata/response.")
                    else:
                        for idx, source in enumerate(sources):
                            # Normalize
                            if isinstance(source, str):
                                content = source
                                meta_info = {}
                            else:
                                content = source.get("content") or source.get("text") or "No Content"
                                meta_info = source.get("metadata", {}) or {}

                            chunktype = (
                                g(source, "chunktype", "chunk_type", default=None)
                                or g(meta_info, "chunktype", "chunk_type", default="unknown")
                            )
                            score = (
                                g(source, "score", default=None)
                                or g(meta_info, "score", default="N/A")
                            )
                            parent_id = (
                                g(source, "id", "passage_id", "passageid", default=None)
                                or g(meta_info, "id", "passage_id", "passageid", default=None)
                                or "unknown"
                            )

                            chunk_label = f"Chunk {idx+1} — type={chunktype}, parent_id={parent_id}, score={score}"
                            st.markdown(f"**{chunk_label}**")

                            st.text_area(
                                label=f"Content {idx+1}",
                                value=content,
                                height=150,
                                key=f"tab7_chunk_{i}_{idx}",
                                disabled=True,
                            )

                            with st.popover(f"Metadata {idx+1}"):
                                st.json(source if isinstance(source, dict) else meta_info)

                            # Child hits + expansions (use popovers, not expanders)
                            if isinstance(source, dict):
                                matching = (
                                    source.get("matchingchildren")
                                    or source.get("matching_children")
                                    or meta_info.get("matchingchildren")
                                    or meta_info.get("matching_children")
                                    or []
                                )

                                exp = source.get("expansion") or meta_info.get("expansion") or {}
                                from_children = exp.get("fromchildren") or exp.get("from_children") or []

                                if matching or from_children:
                                    count = len(matching) if matching else len(from_children)
                                    with st.popover(f"Child chunks retrieved initially ({count})"):
                                        if matching:
                                            for j, c in enumerate(matching):
                                                child_id = g(c, "childid", "child_id", default="unknown")
                                                child_score = g(c, "score", default="N/A")
                                                child_text = (
                                                    g(c, "text", default="")
                                                    or g(c, "textpreview", "text_preview", default="")
                                                )

                                                st.markdown(f"- {child_id} | score={child_score}")

                                                if child_text:
                                                    st.text_area(
                                                        label=f"Matched child text {idx+1}.{j+1}",
                                                        value=child_text,
                                                        height=220,
                                                        key=f"tab7_child_full_{i}_{idx}_{j}",
                                                        disabled=True,
                                                    )
                                        else:
                                            for child_id in from_children:
                                                st.write(f"- {child_id}")

                                # FULL CHILD CHUNKS (debug only)
                                if show_all_children:
                                    child_chunks = (
                                        source.get("child_chunks")
                                        or source.get("childchunks")
                                        or meta_info.get("child_chunks")
                                        or meta_info.get("childchunks")
                                        or []
                                    )

                                    if child_chunks:
                                        with st.popover(f"All child chunks under this parent ({len(child_chunks)})"):
                                            for j, ch in enumerate(child_chunks):
                                                if not isinstance(ch, dict):
                                                    continue
                                                ch_id = g(ch, "id", "chunk_id", "chunkid", default=f"child_{j+1}")
                                                ch_text = g(ch, "text", "page_content", "content", default="")
                                                ch_meta = ch.get("metadata", {}) or {}

                                                st.markdown(f"- {ch_id}")
                                                if ch_text:
                                                    st.text_area(
                                                        label=f"Child text {idx+1}.{j+1}",
                                                        value=ch_text,
                                                        height=140,
                                                        key=f"tab7_child_all_{i}_{idx}_{j}",
                                                        disabled=True,
                                                    )
                                                with st.popover(f"Child metadata {idx+1}.{j+1}"):
                                                    st.json(ch_meta)

                            st.divider()
                else:
                    st.caption(
                        "Intermediate retrieval results from this step are merged "
                        "into the final context shown for the last search above."
                    )

        # 3) LLM CONTEXT PREVIEW (no nested expander needed)
        st.header("🧠 Context Passed to LLM")
        compose_step = next((a for a in agents_trace if (a.get("agent") or "").lower() == "compose"), None)

        if compose_step:
            st.success("Found composition step. Showing full input state for the LLM.")
            st.json(compose_step.get("input", {}) or {})
        else:
            st.warning("No composition step found in the trace.")



# Footer
st.divider()
st.caption("NCERT AI Tutor v0.12.0 | Built with Streamlit, FastAPI, LangGraph, and FAISS")
