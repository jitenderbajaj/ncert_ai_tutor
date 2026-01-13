# FILE: backend/agent/engagement.py
"""
Engagement coordinator (LLM-powered)
"""
import json
import logging
from typing import Any, Dict, List, Optional

from backend.providers.registry import get_provider_registry
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _compact_history(
    chat_history: List[Dict[str, str]],
    max_turns: int = 8,
    max_chars: int = 400
) -> List[Dict[str, str]]:
    hist = chat_history or []
    out: List[Dict[str, str]] = []
    for m in hist[-max_turns:]:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            out.append({"role": role, "content": content[:max_chars]})
    return out


def _compact_sources(
    sources: List[Dict[str, Any]],
    max_items: int = 6,
    max_chars: int = 900
) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for d in (sources or [])[:max_items]:
        if not isinstance(d, dict):
            continue

        meta = d.get("metadata") or {}

        text = (
            d.get("page_content")
            or d.get("pagecontent")
            or d.get("page_content".replace("_", ""))  # harmless extra guard
            or d.get("text")
            or d.get("content")
            or ""
        )

        chunk_id = (
            meta.get("chunk_id") or meta.get("chunkid")
            or meta.get("passage_id") or meta.get("passageid")
            or meta.get("id")
        )

        # "One step further": add grounding identifiers (retrieval attaches these in metadata).
        bookid = meta.get("bookid") or d.get("bookid")
        chapterid = meta.get("chapterid") or d.get("chapterid")
        indextype = meta.get("indextype") or d.get("indextype")

        compact.append({
            "chunk_id": chunk_id,
            "bookid": bookid,
            "chapterid": chapterid,
            "indextype": indextype,  # "detail" / "summary"
            "page": meta.get("page") or meta.get("page_number"),
            "source": meta.get("source") or meta.get("file") or meta.get("doc_id"),
            "text": text[:max_chars],
        })

    return compact


def _build_grounding_label(src: List[Dict[str, Any]]) -> str:
    """
    Deterministic, non-LLM grounding label built from the first source.
    If empty, the prompt will instruct the LLM to omit the label.
    """
    if not src:
        return ""

    s0 = src[0] or {}
    parts: List[str] = []

    # Book/Chapter/(IndexType)
    b = s0.get("bookid")
    c = s0.get("chapterid")
    it = s0.get("indextype")

    bci = "/".join([x for x in [b, c] if x])
    if bci:
        parts.append(f"{bci} ({it})" if it else bci)
    elif it:
        parts.append(f"({it})")

    # chunk + page
    chunk_id = s0.get("chunk_id")
    if chunk_id:
        parts.append(f"chunk {chunk_id}")

    page = s0.get("page")
    if page is not None and str(page).strip() != "":
        parts.append(f"page {page}")

    if not parts:
        return ""

    # Keep emojis within your max-2-per-intervention rule (only 1 here).
    return "📌 Grounded from: " + " • ".join(parts)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _fallback_payload(question: str, sources_compact: List[Dict[str, Any]]) -> Dict[str, Any]:
    q1 = "In one sentence, what is the main idea?"
    if sources_compact:
        q1 = "From the source text, state one key definition/process."

    interventions = [
        {
            "position": "inline",
            "type": "micro_quiz",
            "content": (
                "🧠 Quick Quiz (2 mins)\n"
                f"1) {q1}\n"
                "2) 🤔 What part felt confusing?\n\n"
                "🎯 Score yourself: 0/2, 1/2, 2/2.\n"
                "🔥 Streak challenge: get 2/2 twice in a row!"
            ),
        },
        {
            "position": "inline",
            "type": "choose_path",
            "content": (
                "🧭 Choose your next step:\n"
                "A) ⚡ 30-sec recap\n"
                "B) 🌍 One real-life example\n"
                "C) 🧩 A challenge question"
            ),
        },
    ]

    return {
        "interventions": interventions,
        "boredom_meta": {"detected": False, "score": 0.0, "reasons": ["fallback"]},
        "provider_meta": {
            "provider": "fallback",
            "model": "none",
            "router_reason": "fallback",
            "duration_ms": 0,
            "mode": settings.llm_mode,
        },
    }


def _validate_and_normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
    boredom = parsed.get("boredom") or {}
    interventions = parsed.get("interventions") or []

    fixed_interventions: List[Dict[str, Any]] = []
    for it in interventions:
        if not isinstance(it, dict):
            continue
        content = (it.get("content") or "").strip()
        if not content:
            continue
        fixed_interventions.append({
            "position": "inline",
            "type": (it.get("type") or "hints"),
            "content": content,
        })

    if not fixed_interventions:
        raise ValueError("No valid interventions from LLM")

    reasons = boredom.get("reasons", [])
    boredom_meta = {
        "detected": bool(boredom.get("detected", False)),
        "score": _safe_float(boredom.get("score", 0.0), 0.0),
        "reasons": reasons if isinstance(reasons, list) else [],
    }

    return {"boredom_meta": boredom_meta, "interventions": fixed_interventions}


def generate_engagement_interventions(
    draft_answer: str,
    question: str,
    chat_history: List[Dict[str, str]],
    sources: List[Dict[str, Any]],
    hots_level: Optional[str],
    correlation_id: str,
) -> Dict[str, Any]:
    registry = get_provider_registry()

    hist = _compact_history(chat_history)
    src = _compact_sources(sources)
    grounding_label = _build_grounding_label(src)

    prompt = f"""
You are the Engagement Agent for an NCERT AI Tutor.

Mission:
- Append engagement interventions at the END of the answer (append-only).
- Detect boredom/low engagement using Chat History + Question + Draft Answer.
- ALWAYS return at least 1 intervention (even if boredom is low).
- Generate a grounded micro-quiz using ONLY Draft Answer + Sources.
- Add light gamification: self-score + streak.
- Emojis are allowed (max 2 per intervention).

Hard rules:
- Do NOT introduce new factual claims beyond Draft Answer and Sources.
- Do NOT rewrite the Draft Answer.
- Output MUST be valid JSON only (no markdown, no extra text).
- Keep interventions concise and student-friendly.
- In the "micro_quiz" intervention content, the FIRST line MUST be exactly:
  {grounding_label}
- If the line above is empty, omit the grounding line entirely (do not invent one).

Inputs:
Question: {question}
HOTS level: {hots_level}

Grounding label (use verbatim if non-empty):
{grounding_label}

Chat History (recent):
{json.dumps(hist, ensure_ascii=False)}

Sources (grounding snippets):
{json.dumps(src, ensure_ascii=False)}

Draft Answer:
{draft_answer}

Return JSON with EXACT schema:
{{
  "boredom": {{"detected": true/false, "score": 0.0, "reasons": ["..."]}},
  "interventions": [
    {{"position":"inline","type":"micro_quiz|choose_path|study_tip|hints","content":"..."}}
  ]
}}

Guidelines:
- If boredom score >= 0.6 OR user seems stuck, return 2-4 interventions.
- Otherwise return 1-2 interventions.
- Ensure at least one intervention is "micro_quiz" and includes:
  - 2 questions
  - self-score line (0/2, 1/2, 2/2)
  - streak challenge line
- If HOTS level is "hard", include one challenge question inside quiz or as a second intervention.
""".strip()

    try:
        result = registry.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=520,
            correlation_id=correlation_id,
        )

        raw = (result.get("text") or "").strip()
        parsed = json.loads(raw)
        normalized = _validate_and_normalize(parsed)

        provider_meta = {
            "provider": result.get("provider", "unknown"),
            "model": result.get("model", "unknown"),
            "router_reason": result.get("router_reason"),
            "duration_ms": result.get("duration_ms", 0),
            "mode": settings.llm_mode,
        }

        return {
            "interventions": normalized["interventions"],
            "boredom_meta": normalized["boredom_meta"],
            "provider_meta": provider_meta,
        }

    except Exception as e:
        logger.warning(f"[{correlation_id}] Engagement LLM failed; using fallback. Error: {e}")
        return _fallback_payload(question=question, sources_compact=src)

