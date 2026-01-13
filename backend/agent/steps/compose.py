# FILE: backend/agent/steps/compose.py
"""
Compose step: Synthesize final answer (Manual Prompting)
"""
import logging
from typing import Dict, Any
from backend.config import get_settings
from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)
settings = get_settings()

async def compose_step(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("STEP: Compose (with Diagrams)")
    
    # ... (Keep context building logic same as before) ...
    plan = state.get("plan", {})
    query = plan.get("original_question") or state.get("question") or state.get("query")
    documents = state.get("documents", [])
    
    context_parts = []
    citation_ids = []
    if documents:
        seen = set()
        for doc in documents:
            # Handle different doc formats
            if isinstance(doc, dict):
                content = doc.get('text', '')
                metadata = doc.get('metadata', {})
            else:
                content = getattr(doc, 'page_content', '')
                metadata = getattr(doc, 'metadata', {})
            
            # --- NEW: collect a citation id ---
            # chunk_id = (
            #     metadata.get("chunk_id")
            #     or metadata.get("id")
            #     or metadata.get("source_id")
            # )
            chunk_id = (
                (doc.get("id") if isinstance(doc, dict) else None)
                or metadata.get("chunk_id")
                or metadata.get("id")
                or metadata.get("source_id")
            )

            if chunk_id and chunk_id not in citation_ids:
                citation_ids.append(chunk_id)
            # -----------------------------------
            
            # images = metadata.get('images', [])
            # Support both doc-level and metadata-level, and legacy key image_anchors
            if isinstance(doc, dict):
                images = (
                    doc.get("images") or doc.get("image_anchors")
                    or metadata.get("images") or metadata.get("image_anchors")
                    or []
                )
            else:
                images = (
                    getattr(doc, "images", None) or getattr(doc, "image_anchors", None)
                    or metadata.get("images") or metadata.get("image_anchors")
                    or []
                )   

            image_tag = ""
            if images:
                img_list = ", ".join([img.get('caption', 'Diagram') for img in images])
                image_tag = f"\n[Available Diagrams: {img_list}]"
            combined = content + image_tag
            if combined and combined not in seen:
                seen.add(combined)
                context_parts.append(combined)
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No specific context found."

    if not query:
        return {"final_answer": "Error: No query provided."}

    # --- Manual Prompt Construction ---
    
    # FIX: Use an f-string (f"...") so variables {context_text} and {query} are injected.
    prompt_text = f"""You are an expert NCERT Tutor. Answer the student's question using ONLY the provided context.

    Context:
    {context_text}

    User Question: "{query}"

    Instructions:
    1. **ANALYZE CONTEXT TYPE:**
       - **If the context is a Table of Contents (TOC) or List of Topics:** 
         Do NOT merely list the chapters. Instead, select **ONE** key concept from the list that seems most fundamental. Provide a brief, engaging 1-2 sentence definition of it, and ask the student if they would like to explore that specific topic further.
       - **If the context is Detailed Text Passages:** 
         Explain the concept clearly and comprehensively using the provided details.

    2. **DIAGRAM CITATION:** The context may contain "[Available Diagrams: ...]" tags. 
       If your explanation relates to a concept shown in these diagrams, explicitly cite them in your text.
       Format: "(see Figure X.Y)" or "(refer to the diagram)".
       Do NOT hallucinate diagrams that are not listed in the context.

    3. If the context is insufficient, admit it.
    4. Use a simplified, teaching tone suitable for a student.
    
    Answer:"""

    try:
        registry = get_provider_registry()
        
        # Log intent (informational only)
        logger.info("Generating answer via Registry Routing")
        
        # FIX: Call registry.generate() correctly
        # - Passes interpolation-ready prompt
        # - Removes invalid 'provider' argument
        # - Includes correlation_id for tracing
        response_data = registry.generate(
            prompt=prompt_text,
            temperature=0.7,
            max_tokens=1000,
            # correlation_id=state.get("metadata", {}).get("correlation_id")
            correlation_id=state.get("correlation_id")
        )
        
        final_text = response_data.get("text", "")
        # provider_used = response_data.get("provider", "unknown")
        
        return {
            "final_answer": final_text,
            "citations": citation_ids,
            # "metadata": {"provider": provider_used}
        }

    except Exception as e:
        logger.error(f"Composition failed: {e}", exc_info=True)
        return {"final_answer": "Error generating response.", "error": str(e)}
