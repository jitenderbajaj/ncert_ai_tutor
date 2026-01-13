# FILE: backend/services/ingestion_dual.py
"""
Dual-index ingestion with Multi-Index/Hybrid RAG + Parent-Child + Text Cleaning + Metadata Tracking + Image Binding

Complete Features:
1. PDF text extraction with intelligent noise filtering
2. Parent-child document structure (Small-to-Large retrieval)
3. Detail index: Child chunks (512 chars) in FAISS
4. Summary index: LLM-generated summaries (optimized prompt for density)
5. Separate metadata tracking (provider, seed, timestamp)
6. Image extraction and binding to chunks
7. Comprehensive manifests with checksums

Version: 0.11.3
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import json
import hashlib
from datetime import datetime
import re

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from backend.config import get_settings
from backend.multimodal.image_extract import extract_images_from_pdf
from backend.providers.registry import get_provider_registry
from backend.services.text_cleaner import clean_text_for_llm_summary, get_text_statistics

logger = logging.getLogger(__name__)
settings = get_settings()

def ingest_chapter_dual(
    pdf_path: str,
    book_id: str,
    chapter_id: str,
    seed: int = 42,
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Complete dual-index ingestion pipeline with metadata tracking and image binding.
    Same pipeline as before, but now optionally emits progress events for SSE.
    """
    def _emit(event: Dict[str, Any]) -> None:
        if not emit:
            return
        try:
            emit(event)
        except Exception:
            # Never let UI progress reporting break ingestion
            return

    def _progress(pct: int, message: str, step: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "type": "ingest_progress",
            "pct": int(pct),
            "message": message,
            "step": step,
            "book_id": book_id,
            "chapter_id": chapter_id,
        }
        if extra:
            payload.update(extra)
        _emit(payload)

    logger.info("╔════════════════════════════════════════════════════════")
    logger.info(f"║ INGESTION PIPELINE: {book_id}/{chapter_id}")
    logger.info(f"║ Seed: {seed} | Version: 0.11.3")
    logger.info("╚════════════════════════════════════════════════════════")

    np.random.seed(seed)

    _emit({
        "type": "ingest_started",
        "pct": 0,
        "message": "Ingestion started",
        "book_id": book_id,
        "chapter_id": chapter_id,
        "seed": seed,
    })

    # Step 1: Extract and clean text
    logger.info("[1/9] Extracting and cleaning PDF text")
    _progress(5, "Extracting and cleaning PDF text", "extract_clean_text")

    text, cleaning_stats = extract_and_clean_text_from_pdf(pdf_path)

    _progress(
        12,
        "PDF text extracted and cleaned",
        "extract_clean_text_done",
        extra={
            "raw_chars": cleaning_stats.get("raw_chars"),
            "cleaned_chars": cleaning_stats.get("cleaned_chars"),
            "reduction_pct": cleaning_stats.get("reduction_pct"),
        },
    )
    logger.info(f"✓ Text: {len(text):,} chars (reduced {cleaning_stats['reduction_pct']:.1f}%)")

    # TOC extraction (Increment 11.2)
    _progress(15, "Extracting TOC metadata", "toc_extract")
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    toc_structure = _extract_toc_from_text(text)
    try:
        with open(shard_dir / "toc.json", "w", encoding="utf-8") as f:
            json.dump(toc_structure, f, indent=2)
        _progress(18, f"TOC saved ({len(toc_structure)} headings)", "toc_saved", extra={"toc_headings": len(toc_structure)})
        logger.info(f"✓ Extracted {len(toc_structure)} headings for TOC metadata")
    except Exception as e:
        _emit({
            "type": "ingest_warning",
            "pct": 18,
            "message": f"Failed to save TOC metadata: {e}",
            "step": "toc_save_failed",
            "book_id": book_id,
            "chapter_id": chapter_id,
        })
        logger.error(f"Failed to save TOC metadata: {e}")

    # Step 2: Build parent-child structure
    logger.info("[2/9] Building parent-child structure")
    _progress(25, "Building parent-child structure", "parent_child_build")

    parents, children, parent_map = build_parent_child_chunks(
        text=text,
        parent_size=settings.parent_doc_size,
        child_size=settings.child_chunk_size,
        child_overlap=settings.child_chunk_overlap,
        seed=seed,
    )

    _progress(
        35,
        "Parent-child structure built",
        "parent_child_built",
        extra={"num_parents": len(parents), "num_children": len(children)},
    )
    logger.info(f"✓ Parents: {len(parents)}, Children: {len(children)}")

    # Step 3: Save parent documents
    logger.info("[3/9] Saving parent documents")
    _progress(40, "Saving parent documents", "parents_save")
    save_parent_documents(parents, book_id, chapter_id)
    _progress(45, "Parent documents saved", "parents_saved")

    # Step 4: Save parent-child mapping
    logger.info("[4/9] Saving parent-child mapping")
    _progress(48, "Saving parent-child mapping", "parentmap_save")
    save_parent_mapping(parent_map, book_id, chapter_id)
    _progress(50, "Parent-child mapping saved", "parentmap_saved")

    # Step 5: Build detail index
    logger.info("[5/9] Building detail index from child chunks")
    _progress(55, "Building detail FAISS index", "detail_index_build")

    detail_manifest = build_index(
        chunks=children,
        book_id=book_id,
        chapter_id=chapter_id,
        index_type="detail",
        seed=seed,
    )

    _progress(
        65,
        "Detail index built",
        "detail_index_built",
        extra={"detail_num_chunks": (detail_manifest or {}).get("stats", {}).get("num_chunks")},
    )

    # Step 6: Generate LLM summary + metadata
    logger.info("[6/9] Generating LLM summary")
    _progress(70, "Generating LLM summary", "summary_generate")

    summary_result = generate_chapter_summary_with_llm(
        chapter_text=text,
        book_id=book_id,
        chapter_id=chapter_id,
        seed=seed,
    )

    clean_summary = summary_result.get("summary", "")
    summary_metadata = summary_result.get("metadata", {})

    _progress(
        78,
        "LLM summary generated",
        "summary_generated",
        extra={
            "summary_chars": len(clean_summary),
            "generated_by": summary_metadata.get("generated_by"),
            "generation_status": summary_metadata.get("generation_status"),
        },
    )
    logger.info(f"✓ Summary: {len(clean_summary):,} chars by {summary_metadata.get('generated_by')}")

    # Step 7: Build summary index
    logger.info("[7/9] Building summary index")
    _progress(82, "Building summary FAISS index", "summary_index_build")

    summary_chunks = create_summary_chunks(
        summary_text=clean_summary,
        book_id=book_id,
        chapter_id=chapter_id,
        max_chunk_size=getattr(settings, "summary_chunk_threshold", 8000),
    )

    summary_manifest = build_index(
        chunks=summary_chunks,
        book_id=book_id,
        chapter_id=chapter_id,
        index_type="summary",
        seed=seed,
    )

    _progress(
        88,
        "Summary index built",
        "summary_index_built",
        extra={"summary_num_chunks": (summary_manifest or {}).get("stats", {}).get("num_chunks")},
    )

    # Step 8: Extract images
    logger.info("[8/9] Extracting images from PDF")
    _progress(90, "Extracting images from PDF", "images_extract")

    images = extract_images_from_pdf(pdf_path, book_id, chapter_id)

    _progress(92, f"Images extracted ({len(images)})", "images_extracted", extra={"image_count": len(images)})

    # Step 9: Bind images to chunks
    logger.info("[9/9] Binding images to chunks")
    _progress(95, "Binding images to chunks", "images_bind")

    # Bind to both detail and summary chunks if helpers exist
    try:
        if images:
            children = bind_images_to_chunks(children, images, book_id, chapter_id)
            summary_chunks = bind_images_to_chunks(summary_chunks, images, book_id, chapter_id)
        _progress(98, "Images bound to chunks", "images_bound")
    except Exception as e:
        _emit({
            "type": "ingest_warning",
            "pct": 98,
            "message": f"Image binding skipped/failed: {e}",
            "step": "images_bind_failed",
            "book_id": book_id,
            "chapter_id": chapter_id,
        })

    detail_count = (detail_manifest or {}).get("stats", {}).get("num_chunks", 0)
    summary_count = (summary_manifest or {}).get("stats", {}).get("num_chunks", 0)
    image_count = len(images or [])

    result: Dict[str, Any] = {
        "status": "success",
        "book_id": book_id,
        "chapter_id": chapter_id,
        "seed": seed,
        "toc": toc_structure,
        "parent_child_stats": {     
        "num_parents": len(parents),
        "num_children": len(children),
        "avg_children_per_parent": round(len(children) / len(parents), 2) if parents else 0
        },
        "detail_manifest": detail_manifest,
        "summary_manifest": summary_manifest,
        "text_cleaning_stats": cleaning_stats,
        "summary_metadata": summary_metadata,
        "summary_text": clean_summary,
        "images": images,
        "detail_count": detail_count,
        "summary_count": summary_count,
        "image_count": image_count,
    }

    _emit({
        "type": "ingest_complete",
        "pct": 100,
        "message": "Ingestion complete",
        "step": "complete",
        "book_id": book_id,
        "chapter_id": chapter_id,
        "result": result,
    })

    return result


def bind_images_to_chunks(
    chunks: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    book_id: str,
    chapter_id: str
) -> List[Dict[str, Any]]:
    """
    Bind extracted images to their related chunks based on page proximity.
    
    Strategy:
    1. Group images by page number
    2. Estimate page range for each chunk (based on position in text)
    3. Assign nearby images to chunks (±1 page tolerance)
    4. Limit to 2 images per chunk to avoid clutter
    
    Args:
        chunks: List of chunk dicts (children or summary)
        images: List of image dicts from extract_images_from_pdf
        book_id: Book identifier
        chapter_id: Chapter identifier
    
    Returns:
        Updated chunks with image_anchors populated
    """
    if not images:
        logger.debug("[IMAGE BINDING] No images to bind")
        return chunks
    
    logger.info(f"[IMAGE BINDING] Binding {len(images)} images to {len(chunks)} chunks")
    
    # Group images by page
    images_by_page = {}
    for img in images:
        page = img.get("page", 0)
        if page not in images_by_page:
            images_by_page[page] = []
        images_by_page[page].append(img)
    
    logger.debug(f"[IMAGE BINDING] Images across {len(images_by_page)} pages")
    
    # Calculate chunks per page (heuristic)
    total_pages = max(images_by_page.keys()) if images_by_page else 1
    chunks_per_page = max(1, len(chunks) // total_pages)
    
    bound_count = 0
    
    for chunk_idx, chunk in enumerate(chunks):
        # Estimate page for this chunk
        estimated_page = (chunk_idx // chunks_per_page) + 1
        
        # Find images near this page (±1 page tolerance)
        nearby_images = []
        for page_offset in range(-1, 2):  # Check page-1, page, page+1
            check_page = estimated_page + page_offset
            if check_page in images_by_page:
                nearby_images.extend(images_by_page[check_page])
        
        if nearby_images:
            # Add image anchors to chunk (max 2 images per chunk)
            chunk["image_anchors"] = [
                {
                    "image_id": img.get("id"),
                    "image_path": img.get("path"),
                    "filename": img.get("filename"),
                    "page": img.get("page"),
                    "relevance": "nearby"  # Could use semantic similarity in future
                }
                for img in nearby_images[:2]  # Limit to 2 images per chunk
            ]
            bound_count += len(chunk["image_anchors"])
    
    logger.info(f"[IMAGE BINDING] ✓ Bound {bound_count} image anchors to chunks")
    
    return chunks


def extract_and_clean_text_from_pdf(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract and clean text from PDF with statistics"""
    import fitz  # PyMuPDF
    
    logger.debug(f"[EXTRACT] Opening: {pdf_path}")
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for page_num, page in enumerate(doc):
        text_parts.append(page.get_text())
    
    doc.close()
    
    raw_text = "\n\n".join(text_parts)
    stats_before = get_text_statistics(raw_text)
    
    # Clean text
    cleaned_text = clean_text_for_llm_summary(
        text=raw_text,
        aggressive=settings.text_cleaning_aggressive,
        preserve_structure=settings.text_cleaning_preserve_structure
    )
    
    stats_after = get_text_statistics(cleaned_text)
    reduction_pct = (1 - len(cleaned_text) / len(raw_text)) * 100 if len(raw_text) > 0 else 0
    
    cleaning_stats = {
        "raw_chars": len(raw_text),
        "cleaned_chars": len(cleaned_text),
        "reduction_pct": round(reduction_pct, 2),
        "raw_words": stats_before["total_words"],
        "cleaned_words": stats_after["total_words"]
    }
    
    logger.info(f"[CLEAN] ✓ {len(cleaned_text):,} chars ({reduction_pct:.1f}% reduction)")
    
    return cleaned_text, cleaning_stats

def _extract_toc_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Robust TOC Extraction for NCERT PDFs.
    
    Capabilities:
    - Inline Numbered: "1.2.2 Decomposition Reaction" (Single/Double spaces)
    - Split Numbered: "1.1" (Line 1) -> "Chemical Equations" (Line 2)
    - CAPS Headings: "SUMMARY", "WHAT YOU HAVE LEARNT"
    - Artifact Filtering: Removes "AL REACTIONS", "Activity 1.2", etc.
    - Frequency Protection: Keeps numbered sections even if title repeats in text.
    """
    toc = []
    lines = text.split('\n')
    
    # Matches: "1.1 Title", "1.2.2 Title", "1.3 LONG TITLE..."
    # \s+ handles single or multiple spaces (e.g. "1.2.3  Displacement")
    pattern_inline = re.compile(r'^(\d+(?:\.\d+)+)\.?\s+(.*)$')
    
    # Matches: "1.1", "1.3" (Number alone -> Split Line case)
    pattern_number_only = re.compile(r'^(\d+(?:\.\d+)+)\.?\s*$')
    
    # CAPS for "SUMMARY", "WHAT YOU HAVE LEARNT" (No number)
    pattern_caps = re.compile(r'^([A-Z][A-Z\s\-\:]{3,100})$')

    candidates = []
    title_counts = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if not line: continue
        
        # 1. Skip Artifacts immediately
        if "Activity" in line or "Figure" in line or "Reprint" in line or "Table" in line: 
            continue

        # 2. Check Inline Numbered (1.1 Title)
        match = pattern_inline.match(line)
        if match:
            num_str = match.group(1)
            title = match.group(2).strip()
            level = num_str.count('.') + 1
            
            # Validation: Title must look like a header
            # Allow digits at start of title (e.g. "10. Light")
            if len(title) > 2 and (title[0].isupper() or title[0].isdigit()):
                 candidates.append({"level": level, "title": title, "source": "numbered"})
                 title_counts[title] = title_counts.get(title, 0) + 1
            continue

        # 3. Check Split Numbered (1.1 \n Title)
        match_num = pattern_number_only.match(line)
        if match_num and i < len(lines):
            # Peek at next line
            next_line = lines[i].strip()
            # If next line looks like a title (Starts with Cap, not too long)
            if next_line and len(next_line) > 2 and next_line[0].isupper():
                num_str = match_num.group(1)
                level = num_str.count('.') + 1
                candidates.append({"level": level, "title": next_line, "source": "numbered"})
                title_counts[next_line] = title_counts.get(next_line, 0) + 1
                i += 1 # Consume next line
                continue

        # 4. Check Unnumbered CAPS (e.g. SUMMARY)
        if pattern_caps.match(line):
             if "NCERT" in line or "CHAPTER" in line: continue
             if len(line) < 5: continue # Skip short fragments like "ACID"
             
             candidates.append({"level": 1, "title": line, "source": "caps"})
             title_counts[line] = title_counts.get(line, 0) + 1

    # --- Post Processing & Deduplication ---
    final_toc = []
    seen = set()
    
    for item in candidates:
        title = item["title"]
        source = item["source"]
        
        # Cleaning specific artifacts
        if title.endswith(" REA"): title = title.replace(" REA", " REACTIONS")
        
        # Explicit Artifact Filters
        if title in ["CTIONS", "AL REACTIONS", "TYPES OF CHEMIC"]: continue
        if title in seen: continue
        
        # CRITICAL FIX: Frequency Filtering
        # Only filter CAPS headings if they appear frequently (likely Page Headers)
        # NEVER filter Numbered headings (e.g. "Decomposition Reaction")
        if source == "caps" and title_counts.get(title, 0) > 4: 
            continue
        
        item["title"] = title
        final_toc.append(item)
        seen.add(title)

    return final_toc

def build_parent_child_chunks(
    text: str,
    parent_size: int,
    child_size: int,
    child_overlap: int,
    seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build parent-child document structure"""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    parents = []
    children = []
    parent_map = []
    parent_id_counter = 0
    child_id_counter = 0
    
    current_parent_sentences = []
    current_parent_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence)
        
        if current_parent_length + sentence_length > parent_size and current_parent_sentences:
            # Finalize parent
            parent_text = " ".join(current_parent_sentences)
            parent_doc = {
                "id": f"parent_{parent_id_counter}",
                "text": parent_text,
                "size": len(parent_text),
                "metadata": {
                    "parent_index": parent_id_counter,
                    "sentence_count": len(current_parent_sentences)
                }
            }
            parents.append(parent_doc)
            
            # Create children from parent
            parent_children = create_children_from_parent(
                parent_text=parent_text,
                parent_id=f"parent_{parent_id_counter}",
                child_size=child_size,
                child_overlap=child_overlap,
                child_id_offset=child_id_counter
            )
            
            for child_chunk in parent_children:
                children.append(child_chunk)
                parent_map.append({
                    "child_id": child_chunk["id"],
                    "parent_id": f"parent_{parent_id_counter}",
                    "child_index": child_id_counter,
                    "parent_index": parent_id_counter
                })
                child_id_counter += 1
            
            parent_id_counter += 1
            current_parent_sentences = [sentence]
            current_parent_length = sentence_length
        else:
            current_parent_sentences.append(sentence)
            current_parent_length += sentence_length
    
    # Final parent
    if current_parent_sentences:
        parent_text = " ".join(current_parent_sentences)
        parent_doc = {
            "id": f"parent_{parent_id_counter}",
            "text": parent_text,
            "size": len(parent_text),
            "metadata": {"parent_index": parent_id_counter, "sentence_count": len(current_parent_sentences)}
        }
        parents.append(parent_doc)
        
        parent_children = create_children_from_parent(
            parent_text, f"parent_{parent_id_counter}", child_size, child_overlap, child_id_counter
        )
        
        for child_chunk in parent_children:
            children.append(child_chunk)
            parent_map.append({
                "child_id": child_chunk["id"],
                "parent_id": f"parent_{parent_id_counter}",
                "child_index": len(children) - 1,
                "parent_index": parent_id_counter
            })
    
    logger.info(f"[PARENT-CHILD] ✓ {len(parents)} parents, {len(children)} children")
    return parents, children, parent_map


def create_children_from_parent(
    parent_text: str,
    parent_id: str,
    child_size: int,
    child_overlap: int,
    child_id_offset: int
) -> List[Dict[str, Any]]:
    """Create overlapping child chunks from parent"""
    
    sentences = re.split(r'(?<=[.!?])\s+', parent_text)
    children = []
    current_chunk_sentences = []
    current_chunk_length = 0
    local_child_idx = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence)
        
        if current_chunk_length + sentence_length > child_size and current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            child_chunk = {
                "id": f"child_{child_id_offset + local_child_idx}",
                "text": chunk_text,
                "parent_id": parent_id,
                "passage_id": f"child_{child_id_offset + local_child_idx}",
                "chunk_size": len(chunk_text),
                "overlap": child_overlap,
                "type": "child_chunk",
                "image_anchors": [],  # Will be populated by bind_images_to_chunks
                "metadata": {"local_index": local_child_idx}
            }
            children.append(child_chunk)
            local_child_idx += 1
            
            # Overlap
            num_overlap_sentences = max(1, child_overlap // 50)
            overlap_sentences = current_chunk_sentences[-num_overlap_sentences:] if child_overlap > 0 else []
            current_chunk_sentences = overlap_sentences + [sentence]
            current_chunk_length = sum(len(s) for s in current_chunk_sentences)
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_length
    
    # Final child
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        children.append({
            "id": f"child_{child_id_offset + local_child_idx}",
            "text": chunk_text,
            "parent_id": parent_id,
            "passage_id": f"child_{child_id_offset + local_child_idx}",
            "chunk_size": len(chunk_text),
            "overlap": child_overlap,
            "type": "child_chunk",
            "image_anchors": [],  # Will be populated by bind_images_to_chunks
            "metadata": {"local_index": local_child_idx}
        })
    
    return children


def save_parent_documents(parents: List[Dict[str, Any]], book_id: str, chapter_id: str):
    """Save parent documents to parents.jsonl"""
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    with open(shard_dir / "parents.jsonl", 'w', encoding='utf-8') as f:
        for parent in parents:
            f.write(json.dumps(parent, ensure_ascii=False) + '\n')


def save_parent_mapping(parent_map: List[Dict[str, Any]], book_id: str, chapter_id: str):
    """Save parent-child mapping to parent_map.jsonl"""
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    with open(shard_dir / "parent_map.jsonl", 'w', encoding='utf-8') as f:
        for mapping in parent_map:
            f.write(json.dumps(mapping, ensure_ascii=False) + '\n')


def generate_chapter_summary_with_llm(
    chapter_text: str,
    book_id: str,
    chapter_id: str,
    seed: int
) -> Dict[str, Any]:
    """
    Generate LLM summary with separate metadata storage.
    
    Returns:
        {
            "summary": str,      # Clean summary text (for indexing)
            "metadata": dict     # Provider, seed, timestamp (stored separately)
        }
    """
    logger.info(f"[LLM SUMMARY] Generating for {book_id}/{chapter_id}")
    
    registry = get_provider_registry()
    
    # Truncate if needed
    max_input_chars = settings.summary_max_input_chars
    original_length = len(chapter_text)
    
    if len(chapter_text) > max_input_chars:
        truncated = chapter_text[:max_input_chars]
        last_para = truncated.rfind('\n\n')
        if last_para > max_input_chars * 0.8:
            chapter_text = truncated[:last_para]
        else:
            chapter_text = truncated
        logger.warning(f"[LLM SUMMARY] Truncated: {original_length:,} → {len(chapter_text):,} chars")
    
    # OPTIMIZED PROMPT - Dense, single-paragraph format
    prompt = f"""Summarize this NCERT chapter in ONE dense paragraph of 400-500 words.

Book: {book_id} | Chapter: {chapter_id}

STRICT FORMAT:
- Write as ONE continuous paragraph (no sections, headers, or bullets)
- Direct factual content - NO meta-commentary like "this chapter covers" or "students will learn"
- Cover: core concepts, main topics, key definitions
- Skip: specific examples, activities, exercises, questions
- Dense style: every sentence adds new information

WRITE DIRECTLY - Start with the content, not commentary.

Chapter text:
{chapter_text}

Summary paragraph:"""
    
    try:
        response = registry.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=700,  # Reduced for brevity
            correlation_id=f"summary_{book_id}_{chapter_id}"
        )
        
        summary = response["text"].strip()
        provider = response.get("provider", "unknown")
        model = response.get("model", "unknown")
        
        # Clean summary (remove any headers LLM might add)
        summary = remove_markdown_headers(summary)
        summary = remove_meta_commentary(summary)
        
        # Create metadata
        metadata = {
            "chapter_id": chapter_id,
            "book_id": book_id,
            "generated_by": f"{provider}/{model}",
            "provider": provider,
            "model": model,
            "seed": seed,
            "timestamp": datetime.utcnow().isoformat(),
            "input_chars": len(chapter_text),
            "output_chars": len(summary),
            "compression_ratio": round(len(summary) / len(chapter_text), 3),
            "generation_status": "success"
        }
        
        # Save metadata separately
        save_summary_metadata(metadata, book_id, chapter_id)
        
        logger.info(f"[LLM SUMMARY] ✓ {len(summary):,} chars by {provider}/{model}")
        
        return {
            "summary": summary,
            "metadata": metadata
        }
    
    except Exception as e:
        logger.error(f"[LLM SUMMARY] Failed: {e}")
        
        fallback_summary = chapter_text[:1500]
        
        metadata = {
            "chapter_id": chapter_id,
            "book_id": book_id,
            "generated_by": "fallback",
            "provider": "none",
            "model": "none",
            "seed": seed,
            "timestamp": datetime.utcnow().isoformat(),
            "input_chars": len(chapter_text),
            "output_chars": len(fallback_summary),
            "compression_ratio": round(len(fallback_summary) / len(chapter_text), 3),
            "generation_status": "fallback",
            "error": str(e)
        }
        
        save_summary_metadata(metadata, book_id, chapter_id)
        
        return {
            "summary": fallback_summary,
            "metadata": metadata
        }


def save_summary_metadata(metadata: Dict[str, Any], book_id: str, chapter_id: str):
    """Save summary metadata separately from indexed content"""
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = shard_dir / "summary_metadata.json"
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.debug(f"[METADATA] Saved to {metadata_path}")


def remove_markdown_headers(text: str) -> str:
    """Remove markdown headers from summary"""
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)
    return text.strip()


def remove_meta_commentary(text: str) -> str:
    """Remove meta-commentary from summary"""
    meta_patterns = [
        r'^Here is (a|an).*?summary.*?:?\s*',
        r'^(This|The) chapter (introduces|covers|explains|discusses).*?\.\s*',
        r'^In this chapter,.*?\.\s*',
        r'^\*\*.*?\*\*\s*'  # Remove bold headers
    ]
    
    for pattern in meta_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text.strip()

def create_summary_chunks(
    summary_text: str,
    book_id: str,
    chapter_id: str,
    max_chunk_size: int = 8000
) -> List[Dict[str, Any]]:
    """Create summary chunks with configurable threshold"""
    
    summary_length = len(summary_text)
    
    # Single chunk case
    if summary_length <= max_chunk_size:
        logger.info(f"[SUMMARY CHUNKS] Single chunk: {summary_length:,} chars")
        return [{
            "id": f"summary_{chapter_id}_main",
            "text": summary_text,
            "passage_id": f"summary_{chapter_id}_main",
            "chunk_size": summary_length,
            "overlap": 0,
            "type": "llm_generated_summary",
            "image_anchors": [],
            "metadata": {
                "book_id": book_id,
                "chapter_id": chapter_id,
                "chunk_strategy": "single",
                "total_sections": 1
            }
        }]
    
    logger.warning(f"[SUMMARY CHUNKS] Splitting: {summary_length:,} > {max_chunk_size:,}")
    
    # Try splitting by paragraphs first
    sections = summary_text.split('\n\n')
    
    # If no paragraph breaks, split by sentences
    if len(sections) == 1:
        logger.debug("[SUMMARY CHUNKS] No paragraph breaks, splitting by sentences")
        sections = re.split(r'(?<=[.!?])\s+', summary_text)
    
    # If still no splits (very rare), force split by character count
    if len(sections) == 1:
        logger.debug("[SUMMARY CHUNKS] No sentence breaks, forcing character split")
        sections = [summary_text[i:i+max_chunk_size] for i in range(0, len(summary_text), max_chunk_size)]
    
    chunks = []
    current_text = ""
    chunk_idx = 0
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # Check if adding this section exceeds threshold
        test_length = len(current_text) + len(section) + 2
        
        if test_length <= max_chunk_size:
            # Add to current chunk
            if current_text:
                current_text += "\n\n" + section
            else:
                current_text = section
        else:
            # Save current chunk if it has content
            if current_text:
                chunks.append({
                    "id": f"summary_{chapter_id}_sec{chunk_idx}",
                    "text": current_text,
                    "passage_id": f"summary_{chapter_id}_sec{chunk_idx}",
                    "chunk_size": len(current_text),
                    "overlap": 0,
                    "type": "llm_generated_summary_section",
                    "image_anchors": [],
                    "metadata": {
                        "chunk_strategy": "split",
                        "section_index": chunk_idx
                    }
                })
                chunk_idx += 1
            
            # Start new chunk with current section
            current_text = section
    
    # Add final chunk
    if current_text:
        chunks.append({
            "id": f"summary_{chapter_id}_sec{chunk_idx}",
            "text": current_text,
            "passage_id": f"summary_{chapter_id}_sec{chunk_idx}",
            "chunk_size": len(current_text),
            "overlap": 0,
            "type": "llm_generated_summary_section",
            "image_anchors": [],
            "metadata": {
                "chunk_strategy": "split",
                "section_index": chunk_idx
            }
        })
    
    # Add total_sections to all chunks
    total_sections = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["total_sections"] = total_sections
    
    logger.info(f"[SUMMARY CHUNKS] ✓ Split into {total_sections} sections")
    
    return chunks


def build_index(
    chunks: List[Dict[str, Any]],
    book_id: str,
    chapter_id: str,
    index_type: str,
    seed: int
) -> Dict[str, Any]:
    """Build FAISS index with embeddings"""
    
    if not chunks:
        raise ValueError(f"No chunks for {index_type} index")
    
    # Embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype('float32')
    
    # Build FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = shard_dir / f"{index_type}.index"
    chunks_path = shard_dir / f"{index_type}_chunks.jsonl"
    
    faiss.write_index(index, str(index_path))
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    # Save full summary text (clean, no metadata header)
    if index_type == "summary":
        summary_text_path = shard_dir / "chapter_summary.txt"
        with open(summary_text_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join([c["text"] for c in chunks]))
    
    # Checksums
    with open(index_path, 'rb') as f:
        index_checksum = hashlib.sha256(f.read()).hexdigest()
    with open(chunks_path, 'rb') as f:
        chunks_checksum = hashlib.sha256(f.read()).hexdigest()
    
    # Manifest
    manifest = {
        "book_id": book_id,
        "chapter_id": chapter_id,
        "index_type": index_type,
        "shard_version": "0.11.3",
        "created_at": datetime.utcnow().isoformat(),
        "params": {
            "seed": seed,
            "embedding_model": "all-MiniLM-L6-v2",
            "faiss_factory": "IndexFlatL2",
            "summary_generation": "llm" if index_type == "summary" else "none",
            "parent_child_enabled": index_type == "detail",
            "image_binding_enabled": True
        },
        "checksums": {"index": index_checksum, "chunks": chunks_checksum},
        "stats": {
            "num_chunks": len(chunks),
            "embedding_dim": dimension,
            "total_text_length": sum(len(c["text"]) for c in chunks)
        }
    }
    
    with open(shard_dir / f"{index_type}_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[BUILD INDEX] ✓ {index_type}: {len(chunks)} chunks")
    return manifest


# Export public API
__all__ = [
    # Main ingestion
    "ingest_chapter_dual",
    
    # Text processing
    "extract_and_clean_text_from_pdf",
    
    # Parent-child structure
    "build_parent_child_chunks",
    "create_children_from_parent",
    "save_parent_documents",
    "save_parent_mapping",
    
    # LLM summary
    "generate_chapter_summary_with_llm",
    "create_summary_chunks",
    "save_summary_metadata",
    
    # Text cleaning utilities
    "remove_markdown_headers",
    "remove_meta_commentary",
    
    # Image binding (NEW)
    "bind_images_to_chunks",
    
    # Index building
    "build_index"
    
    "_extract_toc_from_text"
]
