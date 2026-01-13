# FILE: backend/agent/engagement.py
"""
Engagement coordinator (LLM-powered)
"""
import json
import logging
import re
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


# def _ends_with_explore_cta(draft_answer: str) -> bool:
#     """
#     Detect if the draft answer already ends with a 'next-step' CTA like
#     'Would you like to explore ... further?'. This is common in TOC-style answers.
#     If true, Engagement should NOT add a competing micro-quiz CTA.
#     """
#     t = (draft_answer or "").strip().lower()
#     if not t:
#         return False

#     tail = t[-350:]  # only check the end
#     patterns = [
#         "would you like to explore",
#         "would you like to learn more",
#         "do you want to explore",
#         "would you like to go deeper",
#         "shall we explore",
#         "would you like to explore this topic further",
#     ]
#     return any(p in tail for p in patterns)


def _ends_with_explore_cta(draft_answer: Optional[str]) -> bool:
    """
    True only if the *ending* of the draft answer contains a next-step CTA.
    Prevents false positives from earlier occurrences of CTA-like phrases.
    """
    if not draft_answer or not isinstance(draft_answer, str):
        return False

    # Normalize
    t = draft_answer.strip()
    if not t:
        return False

    # Look only at the end, but in a way that respects sentence boundaries.
    # Take last ~350 chars, then trim leading partial sentence if possible.
    tail = t[-350:].strip()
    tail_l = tail.lower()

    # Strong anchor: CTA should be in the last 1–2 sentences.
    # We accept common endings like ?, . , ! or end-of-text.
    # Also accept "next step:" / "choose your next step:" blocks.
    anchored_patterns = [
        r"(?:would you like to explore(?: this topic)?(?: further)?)(?:\s+from the chapter)?\s*(?:\?|\.|!|$)",
        r"(?:would you like to learn more)(?:\s+about (?:this|it))?\s*(?:\?|\.|!|$)",
        r"(?:do you want to explore)(?:\s+this)?(?:\s+further)?\s*(?:\?|\.|!|$)",
        r"(?:would you like to go deeper)(?:\s+into (?:this|it))?\s*(?:\?|\.|!|$)",
        r"(?:shall we explore)(?:\s+this)?(?:\s+further)?\s*(?:\?|\.|!|$)",
        r"(?:next step\s*:)\s*(?:\n|$)",
        r"(?:choose your next step\s*:)\s*(?:\n|$)",
    ]

    # Quick structural check: if it ends with a choose-path style CTA (A/B/C),
    # treat it as CTA-present even if phrasing is different.
    # (This is much stricter than "any substring in tail".)
    choose_path_block = bool(
        re.search(r"(?im)^\s*(?:🧭\s*)?choose\s+your\s+next\s+step\s*:?\s*$", tail)
        and re.search(r"(?im)^\s*A\)\s+.+$", tail)
        and re.search(r"(?im)^\s*B\)\s+.+$", tail)
        and re.search(r"(?im)^\s*C\)\s+.+$", tail)
    )

    if choose_path_block:
        return True

    # Sentence-anchored match in the tail
    return any(re.search(pat, tail_l, flags=re.IGNORECASE) for pat in anchored_patterns)



def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# def _fallback_payload(question: str, sources_compact: List[Dict[str, Any]]) -> Dict[str, Any]:
#     q1 = "In one sentence, what is the main idea?"
#     if sources_compact:
#         q1 = "From the source text, state one key definition/process."

#     interventions = [
#         {
#             "position": "inline",
#             "type": "micro_quiz",
#             "content": (
#                 "🧠 Quick Quiz (2 mins)\n"
#                 f"1) {q1}\n"
#                 "2) 🤔 What part felt confusing?\n\n"
#                 "🎯 Score yourself: 0/2, 1/2, 2/2.\n"
#                 "🔥 Streak challenge: get 2/2 twice in a row!"
#             ),
#         },
#         {
#             "position": "inline",
#             "type": "choose_path",
#             "content": (
#                 "🧭 Choose your next step:\n"
#                 "A) ⚡ 30-sec recap\n"
#                 "B) 🌍 One real-life example\n"
#                 "C) 🧩 A challenge question"
#             ),
#         },
#     ]

#     return {
#         "interventions": interventions,
#         "boredom_meta": {"detected": False, "score": 0.0, "reasons": ["fallback"]},
#         "provider_meta": {
#             "provider": "fallback",
#             "model": "none",
#             "router_reason": "fallback",
#             "duration_ms": 0,
#             "mode": settings.llm_mode,
#         },
#     }

def _fallback_payload(
    question: str,
    sources_compact: List[Dict[str, Any]],
    cta_already_present: bool,
    grounding_label: str = "",
) -> Dict[str, Any]:
    q1 = "In one sentence, what is the main idea?"
    if sources_compact:
        q1 = "From the source text, state one key definition/process."

    # CTA_ALREADY_PRESENT == true -> EXACTLY ONE choose_path (inline)
    if cta_already_present:
        interventions = [
            {
                "position": "inline",
                "type": "choose_path",
                "content": (
                    "🧭 Choose your next step:\n"
                    "A) ⚡ 30-sec recap\n"
                    "B) 🌍 One real-life example\n"
                    "C) 🧩 A challenge question"
                ),
            }
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

    # CTA_ALREADY_PRESENT == false -> EXACTLY TWO: 1) mcq(ui) + 2) micro_quiz(inline)
    grounding_line = (grounding_label.strip() + "\n") if grounding_label and grounding_label.strip() else ""

    interventions = [
        {
            "position": "ui",
            "type": "mcq",
            "mcq": {
                "id": "fallback-mcq",
                "question": f"Based on the answer, which option best matches the key idea?",
                "options": [
                    {"key": "A", "text": "Acid + Base → Salt + Water (neutralisation)."},
                    {"key": "B", "text": "Acids and bases do not react with each other."},
                ],
                "answer_key": "A",
                "explanation": "The draft answer is about neutralisation producing salt and water.",
            },
        },
        {
            "position": "inline",
            "type": "micro_quiz",
            "content": (
                f"{grounding_line}"
                "🧠 Quick Quiz (2 mins)\n"
                f"1) {q1}\n"
                "2) 🤔 What part felt confusing?\n\n"
                "🎯 Score yourself: 0/2, 1/2, 2/2.\n"
                "🔥 Streak challenge: get 2/2 twice in a row!"
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


#     return {"boredom_meta": boredom_meta, "interventions": fixed_interventions}
def _validate_and_normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
    boredom = parsed.get("boredom") or {}
    interventions = parsed.get("interventions") or []

    fixed_interventions: List[Dict[str, Any]] = []

    for it in interventions:
        if not isinstance(it, dict):
            continue

        it_type = (it.get("type") or "hints").strip()
        position = (it.get("position") or "inline").strip()

        # --- Case 1: structured MCQ (UI or inline) ---
        if it_type == "mcq":
            mcq = it.get("mcq")
            if not isinstance(mcq, dict):
                continue

            q = (mcq.get("question") or "").strip()
            opts = mcq.get("options") or []
            answer_key = (mcq.get("answer_key") or mcq.get("answerKey") or "").strip()

            if not q or not isinstance(opts, list) or len(opts) < 2 or not answer_key:
                continue

            # Normalize options
            norm_opts = []
            for o in opts:
                if not isinstance(o, dict):
                    continue
                k = (o.get("key") or "").strip()
                t = (o.get("text") or "").strip()
                if k and t:
                    norm_opts.append({"key": k, "text": t})

            if len(norm_opts) < 2:
                continue

            fixed_interventions.append({
                "position": position if position in {"inline", "ui"} else "ui",
                "type": "mcq",
                "mcq": {
                    "id": (mcq.get("id") or "").strip() or None,
                    "question": q,
                    "options": norm_opts,
                    "answer_key": answer_key,
                    "explanation": (mcq.get("explanation") or "").strip(),
                },
            })
            continue

        # --- Case 2: legacy text interventions ---
        content = (it.get("content") or "").strip()
        if not content:
            continue

        fixed_interventions.append({
            "position": "inline",
            "type": it_type,
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

    cta_already_present = _ends_with_explore_cta(draft_answer)

    prompt = f"""
You are the Engagement Agent for an NCERT AI Tutor.

Mission:
- Append engagement interventions at the END of the answer (append-only).
- Detect boredom/low engagement using Chat History + Question + Draft Answer.
- Generate interventions grounded using ONLY Draft Answer + Sources.
- Add light gamification: self-score + streak.
- Emojis are allowed (max 2 per intervention).

Hard rules:
- Do NOT introduce new factual claims beyond Draft Answer and Sources.
- Do NOT rewrite the Draft Answer.
- Output MUST be valid JSON only (no markdown, no extra text).
- Keep interventions concise and student-friendly.
- If you output an MCQ, set position="ui" (never inline).

Grounding label rule (only applies if you produce a micro_quiz):
- In the "micro_quiz" intervention content, the FIRST line MUST be exactly:
  {grounding_label}
- If the line above is empty, omit the grounding line entirely (do not invent one).

Critical UX rule (avoid conflicting CTAs):
- The Draft Answer may already end with a "Would you like to explore further?" question.
- If CTA_ALREADY_PRESENT is true:
  - Return EXACTLY ONE intervention of type "choose_path".
  - Do NOT return a "micro_quiz" or "mcq".
- If CTA_ALREADY_PRESENT is false:
  - Return EXACTLY 2 interventions:
    1) Exactly ONE intervention of type "mcq" with position="ui" (never inline).
    2) Exactly ONE intervention of type "micro_quiz" with position="inline" that includes:
       - 2 questions
       - self-score line (0/2, 1/2, 2/2)
       - streak challenge line
  - Do NOT return "choose_path" in this case.

Inputs:
Question: {question}
HOTS level: {hots_level}
CTA_ALREADY_PRESENT: {str(cta_already_present).lower()}

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
    {{"position":"inline","type":"choose_path","content":"..."}}
  ]
}}

Schema notes:
- If CTA_ALREADY_PRESENT is true: interventions MUST be exactly one choose_path (inline).
- If CTA_ALREADY_PRESENT is false: interventions MUST be exactly two items:
  1) {{"position":"ui","type":"mcq","mcq":{{"id":"optional","question":"...","options":[{{"key":"A","text":"..."}},{{"key":"B","text":"..."}}],"answer_key":"A","explanation":"optional"}}}}
  2) {{"position":"inline","type":"micro_quiz","content":"..."}}
- Do NOT output study_tip or hints.

Guidelines:
- Always obey the Critical UX rule above (it overrides all other guidance).
- If HOTS level is "hard" and CTA_ALREADY_PRESENT is false, make the MCQ a challenge question.
- Keep the micro_quiz short and grounded; do not add any extra interventions beyond what the Critical UX rule allows.
""".strip()


    try:
        result = registry.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=520,
            correlation_id=correlation_id,
        )

        if not isinstance(result, dict):
            raise TypeError(f"registry.generate returned {type(result)}")

        # Extract provider text safely (handles different provider wrapper shapes)
        val = result.get("text")

        if val is None:
            val = result.get("content") or result.get("message") or result.get("output_text")

        if isinstance(val, dict):
            val = val.get("content") or val.get("text") or val.get("value")

        if isinstance(val, list):
            val = "\n".join(str(x) for x in val)

        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValueError("Provider returned empty text payload")

        raw = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
        raw = raw.strip()

        # Parse JSON once, with preview on failure
        try:
            parsed = json.loads(raw)
        except Exception:
            logger.warning(
                f"[{correlation_id}] Engagement JSON parse failed. raw_preview={raw[:400]!r}"
            )
            raise

        if not isinstance(parsed, dict):
            raise ValueError(f"Engagement output JSON must be an object, got {type(parsed)}")

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
            return _fallback_payload(
                question=question,
                sources_compact=src,
                cta_already_present=cta_already_present,
                grounding_label=grounding_label,
            )



