# FILE: backend/agent/steps/retriever.py
"""
Retriever step: fetch relevant passages from dual indices.
Handles both standard retrieval (sync) and comprehensive async summary retrieval.
"""
import logging
from typing import Dict, Any, List

from backend.services.retrieval_dual import retrieve_passages, retrieve_all_chapter_summaries

logger = logging.getLogger(__name__)

async def retriever_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the retrieval strategy defined in the plan.
    
    Supports:
    1. Standard Retrieval (Detail/Summary/Hybrid) for a specific chapter.
    2. Comprehensive Retrieval (All Chapters) for subject-wide queries (Async Tier 4).
    3. Structure Retrieval (TOC).
    4. General Chat (No-Op).
    """
    logger.info("STEP: Retriever")
    
    plan = state.get("plan", {})
    if not plan:
        logger.warning("No plan found in state, skipping retrieval.")
        # return {"retrieved_docs": []}
        return {"documents": []}

    # 1. Extract Strategy FIRST
    strategy = plan.get("strategy", "retrieve")
    
    # --- DEDUPE: avoid fetching TOC twice (retrieve + retrieve_refined) ---
    if strategy == "structure_lookup" and state.get("documents"):
        logger.info("[RETRIEVER] Skipping повтор retrieval for structure_lookup (documents already present).")
        return {"documents": state["documents"]}

    
    # 2. Extract Query
    # query = plan.get("query") or state.get("query")
    # query = plan.get("original_question") or state.get("question") or state.get("query")
    query = plan.get("question") or state.get("question") or state.get("query")
  
    # 3. NOW check both
    if not query and strategy not in ["structure_lookup", "general_chat"]:
        logger.warning("Retriever: No query found. Skipping.")
        return {"documents": []}

    
    # # Extract parameters from Plan
    # query = plan.get("query") or state.get("query")
    # if not query and strategy not in ["structure_lookup", "general_chat"]:
    #     logger.warning("Retrieval skipped: No query provided.")
    #     return {"retrieved_docs": []}
    book_id = plan.get("book_id") or plan.get("bookid")
    chapter_id = plan.get("chapter_id") or plan.get("chapterid")
    
    strategy = plan.get("strategy", "retrieve")
    index_type = plan.get("index_hint") or plan.get("indextype", "detail")
    
    # Expand parents: check both 'expand_to_parents' and 'expandtoparents'
    # expand_parents = plan.get("expand_to_parents", plan.get("expandtoparents", False))
    expand_to_parents_raw = plan.get("expand_to_parents")
    expandtoparents_raw = plan.get("expandtoparents")

    expand_parents = (
        expand_to_parents_raw
        if expand_to_parents_raw is not None
        else (expandtoparents_raw if expandtoparents_raw is not None else False)
    )
    
    # Summary Sampler: check both 'summary_sampler' and 'summarysampler'
    summary_sampler = plan.get("summary_sampler", plan.get("summarysampler", "none"))

    logger.info(
        "[RETRIEVER][DEBUG] plan keys=%s "
        "expand_to_parents=%s expandtoparents=%s => expand_parents=%s "
        "summary_sampler=%s summarysampler=%s => summary_sampler=%s "
        "strategy=%s index_hint=%s book_id=%s chapter_id=%s query_preview=%s",
        sorted(list(plan.keys())),
        expand_to_parents_raw,
        expandtoparents_raw,
        expand_parents,
        plan.get("summary_sampler"),
        plan.get("summarysampler"),
        summary_sampler,
        strategy,
        plan.get("index_hint"),
        book_id,
        chapter_id,
        (query[:80] if isinstance(query, str) else str(query)[:80]),
    )

    # --- DEDUPE: avoid fetching ALL chapter summaries twice ---
    if summary_sampler == "all" and book_id:
        existing_docs = state.get("documents") or []
        if existing_docs and any(
            isinstance(d, dict)
            and d.get("metadata", {}).get("retrieval_strategy") == "summary_sampler_all"
            for d in existing_docs
        ):
            logger.info("[RETRIEVER] Skipping повтор retrieval for summary_sampler=all (documents already present).")
            return {"documents": existing_docs}

    logger.info(
        f"Retrieving for query='{query}' strategy={strategy} book={book_id} ch={chapter_id} "
        f"index={index_type} expand={expand_parents} sampler={summary_sampler}"
    )

    results = []


    try:
        # BRANCH 0A: General Chat (Skip Retrieval)
        if strategy == "general_chat":
            logger.info("Strategy is general_chat. Skipping retrieval.")
            #  return {"retrieved_docs": []}
            return {"documents": []}

        
        elif strategy == "structure_lookup" and book_id:
            logger.info(f"Executing Structure Lookup for {book_id} (Chapter: {chapter_id})")
            from backend.services.retrieval_dual import retrieve_book_structure
            
            # --- DEBUG: Start ---
            try:
                logger.info("[DEBUG] Calling retrieve_book_structure...")
                structure_text = await retrieve_book_structure(
                    book_id=book_id, 
                    chapter_id=chapter_id
                )
                logger.info(f"[DEBUG] structure_text type: {type(structure_text)}")
                logger.info(f"[DEBUG] structure_text length: {len(str(structure_text))}")
                logger.info(f"[DEBUG] structure_text preview: {str(structure_text)[:100]}")
                
                # results = [{
                #     "text": structure_text,
                #     # Dual compatibility keys
                #     "page_content": structure_text, 
                #     # Nested metadata to satisfy graph consumers
                #     "metadata": {
                #         "source": "meta_index",
                #         "type": "structure",
                #         "book_id": book_id,
                #         "chunk_id": f"toc_{book_id}_{chapter_id or 'all'}", # Unique ID
                #         "images": []
                #     }
                # }]
                toc_id = f"toc_{book_id}_{chapter_id or 'all'}"
                results = [{
                    "id": toc_id,
                    "text": structure_text,
                    "page_content": structure_text,   # compat
                    "score": 1.0,
                    "chunk_type": "structure",        # matches your other retrieval docs pattern
                    "metadata": {
                        "source": "meta_index",
                        "type": "structure",          # engagement.py relies on this [file:7]
                        "book_id": book_id,
                        "chapter_id": chapter_id,
                        "chunk_id": toc_id,
                        "index_type": "meta",
                        "images": [],                 # keep empty
                        "image_anchors": [],          # legacy compat
                    }
                }]

                logger.info(f"[DEBUG] Assigned results list with {len(results)} item(s)")
                
            except Exception as inner_e:
                logger.error(f"[DEBUG] CRITICAL ERROR in Structure Block: {inner_e}", exc_info=True)
                results = []
            # --- DEBUG: End ---
      
        # BRANCH 1: Tier 4 - All Chapter Summaries (Subject-Wide)
        elif summary_sampler == "all" and book_id:
            logger.info(f"Executing Summary Sampler: ALL chapters for {book_id}")
            
            top_k = plan.get("top_k_per_chapter", plan.get("topkperchapter", 1))
            
            # Await the async IO function
            # Note: retrieve_all_chapter_summaries returns a list of formatted dicts
            results = await retrieve_all_chapter_summaries(
                book_id=book_id,
                top_k_per_chapter=top_k
            )

        # BRANCH 2: Standard Retrieval (Single Chapter)
        elif book_id and chapter_id:
            logger.info(
                "[RETRIEVER][DEBUG] calling retrieve_passages(index_type=%s, top_k=%s, expand_to_parents=%s)",
                index_type,
                plan.get("top_k", plan.get("topk", 5)),
                expand_parents,
            )

            # Standard retrieval is synchronous
            raw_results = retrieve_passages(
                book_id=book_id,
                chapter_id=chapter_id,
                query=query,
                index_type=index_type,
                top_k=plan.get("top_k", plan.get("topk", 5)),
                expand_to_parents=expand_parents
            )
            
            # Format results to ensure consistent schema
            # results = [{
            #     "text": r["text"],
            #     # FIX: Handle missing 'source' key safely
            #     "source": r.get("source") or r.get("metadata", {}).get("source", "unknown"),
            #     "score": r.get("score", 0.0),
            #     "type": r.get("chunk_type", "unknown"),
            #     "book_id": book_id,
            #     "images": r.get("images", []) 
            # } for r in raw_results]
            # Keep full payload so Tab7 can see matching_children/expansion/child_chunks
            results = []
            for r in raw_results:
                if not isinstance(r, dict):
                    continue

                doc = dict(r)  # shallow copy of everything returned by retrieval_dual

                # Ensure keys used by graph_lg -> sources_for_frontend exist
                doc.setdefault("text", r.get("text", ""))
                doc.setdefault("page_content", r.get("page_content") or doc["text"])

                # Ensure metadata exists (Tab7 + sources_for_frontend rely on this)
                meta = doc.setdefault("metadata", r.get("metadata", {}) or {})
                if not isinstance(meta, dict):
                    meta = {}
                    doc["metadata"] = meta
                meta.setdefault("source", r.get("source") or meta.get("source", "unknown"))
                meta.setdefault("book_id", book_id)
                meta.setdefault("chapter_id", chapter_id)
                # Optional: mirror images into metadata for older consumers
                meta["images"] = doc.get("images") or doc.get("image_anchors") or meta.get("images") or []
                meta["image_anchors"] = doc.get("image_anchors") or meta.get("image_anchors") or meta["images"]
                results.append(doc)
        else:
            # Fallback/Error logging
            if not book_id:
                logger.warning("Missing book_id for retrieval.")
            elif not chapter_id and summary_sampler != "all":
                 logger.warning("Missing chapter_id for single-chapter retrieval.")

    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        # Return empty list allows the agent to proceed rather than crashing
        results = []

    # Update state with results
    # return {"retrieved_docs": results}
    return {"documents": results}


