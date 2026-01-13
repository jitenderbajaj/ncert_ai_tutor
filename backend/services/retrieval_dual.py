# FILE: backend/services/retrieval_dual.py
"""
Dual-index retrieval with parent expansion, metadata access, and comprehensive utilities

Complete Features:
1. Single and batch retrieval with caching
2. Detail index with child → parent expansion
3. Summary index retrieval (LLM-generated summaries)
4. Hybrid retrieval (query both indices)
5. Parent document access by ID
6. Child chunk enumeration for parents
7. Summary metadata access (provider, seed, timestamp)
8. Retrieval statistics and health checks
9. Parent-child consistency validation
10. Cache management

Version: 0.11.2
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import asyncio
import aiofiles

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# from Capstone.Previous_versions.ncert_ai_tutor_i11.backend.services.retrieval_dual import get_child_chunks_for_parent
from backend.config import get_settings
from backend.services.retrieve_cache import RetrieveCache

logger = logging.getLogger(__name__)
settings = get_settings()

_cache = RetrieveCache()


def retrieve_passages(
    query: str,
    book_id: str,
    chapter_id: str,
    index_type: str = "detail",
    top_k: int = 5,
    expand_to_parents: bool = None,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve passages with optional parent expansion.
    
    Retrieval Flow:
    1. Check cache for query
    2. Load FAISS index and chunks
    3. Embed query and search
    4. If detail index + expand_to_parents: expand children → parents
    5. Cache results
    
    Args:
        query: Search query text
        book_id: Book identifier (e.g., "BOOK123")
        chapter_id: Chapter identifier (e.g., "CH1")
        index_type: "detail" (for specific answers) or "summary" (for overviews)
        top_k: Number of results to return
        expand_to_parents: If True, expand children to parents (default: from config)
        use_cache: If True, use retrieval cache
    
    Returns:
        List of passage dicts with text, score, metadata
    """
    if expand_to_parents is None:
        expand_to_parents = settings.retrieve_expand_to_parents
    
    logger.debug(f"[RETRIEVE] query='{query[:50]}...', index={index_type}, top_k={top_k}, expand={expand_to_parents}")
    
    # Check cache
    if use_cache and settings.cache_enabled:
        cached = _cache.get(query, book_id, chapter_id, index_type)
        if cached:
            logger.info(f"[RETRIEVE] ✓ Cache hit: {len(cached)} results")
            return cached
    
    # Load FAISS index and chunks
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    index_path = shard_dir / f"{index_type}.index"
    chunks_path = shard_dir / f"{index_type}_chunks.jsonl"
    
    if not index_path.exists() or not chunks_path.exists():
        logger.warning(f"[RETRIEVE] Index not found: {index_path}")
        return []
    
    index = faiss.read_index(str(index_path))
    
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    logger.debug(f"[RETRIEVE] Loaded {len(chunks)} chunks from {index_type} index")
    
    # Embed query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_numpy=True)[0].astype('float32')
    
    # Search FAISS
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Build chunk results
    chunk_results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(chunks):
            continue
        
        chunk = chunks[idx]
        distance = float(distances[0][i])
        score = 1.0 / (1.0 + distance)
        
        images = chunk.get("images", []) or chunk.get("image_anchors", [])
        
        chunk_results.append({
            "id": chunk.get("id"),
            "text": chunk["text"],
            "score": score,
            "passage_id": chunk.get("passage_id"),
            "parent_id": chunk.get("parent_id"),
            "chunk_type": chunk.get("type", "unknown"),
            "images": images, # <--- CRITICAL FOR DIAGRAMS
            # "metadata": chunk.get("metadata", {})
            "metadata": {
                **chunk.get("metadata", {}),
                "book_id": book_id,
                "chapter_id": chapter_id,
                "chunk_id": chunk.get("id"),        # NEW: stable citation id
                "index_type": index_type,
            },
        })
    
    logger.info(f"[RETRIEVE] Found {len(chunk_results)} matching chunks")
    
    # Parent expansion (only for detail index)
    if expand_to_parents and index_type == "detail":
        logger.info(f"[RETRIEVE] Expanding {len(chunk_results)} children to parents...")
        results = expand_to_parent_documents(chunk_results, book_id, chapter_id)
        logger.info(f"[RETRIEVE] ✓ Expanded to {len(results)} parents")
    else:
        results = chunk_results
    
    # Cache results
    if use_cache and settings.cache_enabled:
        _cache.put(query, book_id, chapter_id, index_type, results)
    
    return results


def retrieve_passages_batch(
    queries: List[str],
    book_id: str,
    chapter_id: str,
    index_type: str = "detail",
    top_k: int = 5,
    expand_to_parents: bool = None,
    use_cache: bool = True
) -> List[List[Dict[str, Any]]]:
    """
    Batch retrieval for multiple queries (efficient for multi-query scenarios).
    
    Use cases:
    - Multi-turn conversations with related queries
    - Batch evaluation and testing
    - Parallel query processing
    - Comparative analysis
    
    Args:
        queries: List of query strings
        book_id: Book identifier
        chapter_id: Chapter identifier
        index_type: "detail" or "summary"
        top_k: Number of results per query
        expand_to_parents: Enable parent expansion
        use_cache: Use retrieval cache
    
    Returns:
        List of result lists (one per query)
        Example: [[results_q1], [results_q2], [results_q3]]
    """
    logger.info(f"[BATCH RETRIEVE] Processing {len(queries)} queries for {book_id}/{chapter_id}")
    
    results = []
    for i, query in enumerate(queries):
        logger.debug(f"[BATCH RETRIEVE] Query {i+1}/{len(queries)}: '{query[:50]}...'")
        
        result = retrieve_passages(
            query=query,
            book_id=book_id,
            chapter_id=chapter_id,
            index_type=index_type,
            top_k=top_k,
            expand_to_parents=expand_to_parents,
            use_cache=use_cache
        )
        results.append(result)
    
    logger.info(f"[BATCH RETRIEVE] ✓ Completed {len(queries)} queries")
    
    return results


def expand_to_parent_documents(
    child_results: List[Dict[str, Any]],
    book_id: str,
    chapter_id: str
) -> List[Dict[str, Any]]:
    """
    Expand child chunks to their parent documents.
    
    Small-to-Large Retrieval Strategy:
    1. Load parent documents from parents.jsonl
    2. Load parent-child mapping from parent_map.jsonl
    3. For each matched child, look up its parent
    4. Aggregate scores by parent (use max child score)
    5. Return parent documents with metadata about matching children
    
    Benefits:
    - Precise search: Small children for semantic matching
    - Rich context: Large parents for LLM composition
    - Deduplication: Each parent returned once even if multiple children match
    
    Args:
        child_results: List of child chunk results from FAISS search
        book_id: Book identifier
        chapter_id: Chapter identifier
    
    Returns:
        List of parent document dicts with aggregated scores
    """
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    
    # Load parents
    parents_path = shard_dir / "parents.jsonl"
    if not parents_path.exists():
        logger.warning(f"[EXPAND] Parents not found at {parents_path}, returning children")
        return child_results
    
    parents_by_id = {}
    with open(parents_path, 'r', encoding='utf-8') as f:
        for line in f:
            parent = json.loads(line)
            parents_by_id[parent["id"]] = parent
    
    logger.debug(f"[EXPAND] Loaded {len(parents_by_id)} parent documents")
    
    # Load mapping
    mapping_path = shard_dir / "parent_map.jsonl"
    if not mapping_path.exists():
        logger.warning(f"[EXPAND] Mapping not found at {mapping_path}, returning children")
        return child_results
    
    child_to_parent = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            mapping = json.loads(line)
            child_to_parent[mapping["child_id"]] = mapping["parent_id"]
    
    logger.debug(f"[EXPAND] Loaded {len(child_to_parent)} parent-child mappings")
    
    # Aggregate by parent
    parent_scores = {}
    parent_children = {}
    parent_images = {}  # <--- NEW: Track images per parent
    
    for child in child_results:
        child_id = child["id"]
        parent_id = child_to_parent.get(child_id)
        
        if not parent_id:
            logger.warning(f"[EXPAND] No parent mapping for child {child_id}")
            continue
        
        # Track best score for this parent
        if parent_id not in parent_scores:
            parent_scores[parent_id] = child["score"]
            parent_children[parent_id] = []
            parent_images[parent_id] = [] # Initialize image list
        else:
            parent_scores[parent_id] = max(parent_scores[parent_id], child["score"])
        
        # Track matching children
        # parent_children[parent_id].append({
        #     "child_id": child_id,
        #     "score": child["score"],
        #     "text_preview": child["text"][:100] + "..." if len(child["text"]) > 100 else child["text"]
        # })

        full_text = child["text"]
        parent_children[parent_id].append({
            "child_id": child_id,
            "score": child["score"],
            "text": full_text,
            "text_preview": full_text[:100] + "..." if len(full_text) > 100 else full_text,
        })

        
        # --- INCREMENT 11.2 UPDATE: Aggregate Images ---
        # Collect images from this child (handle both 'images' and legacy 'image_anchors')
        child_imgs = child.get("images", []) or child.get("image_anchors", [])
        if child_imgs:
            # Avoid duplicates based on image ID
            existing_ids = {img.get("id") for img in parent_images[parent_id]}
            for img in child_imgs:
                img_id = img.get("id")
                if img_id and img_id not in existing_ids:
                    parent_images[parent_id].append(img)
                    existing_ids.add(img_id)

    # Build parent results
    parent_results = []
    for parent_id, score in sorted(parent_scores.items(), key=lambda x: x[1], reverse=True):
        parent_doc = parents_by_id.get(parent_id)
        
        if not parent_doc:
            logger.warning(f"[EXPAND] Parent {parent_id} not found in parents.jsonl")
            continue
        
        # NEW: attach full children (Option A)
        child_chunks = get_child_chunks_for_parent(parent_id, book_id, chapter_id)
        
        # parent_results.append({
        #     "id": parent_id,
        #     "text": parent_doc["text"],
        #     "score": score,
        #     "passage_id": parent_id,
        #     "chunk_type": "parent",
        #     "size": parent_doc.get("size", len(parent_doc["text"])),
        #     "num_matching_children": len(parent_children[parent_id]),
        #     "matching_children": parent_children[parent_id],
            
        #     "child_chunks": child_chunks,
            
        #     # --- INCREMENT 11.2 UPDATE: Pass Aggregated Images ---
        #     "images": parent_images.get(parent_id, []), 
        #     "image_anchors": parent_images.get(parent_id, []), # Legacy compat

        #     "metadata": {
        #         **parent_doc.get("metadata", {}),
        #         "expansion": {
        #             "from_children": [c["child_id"] for c in parent_children[parent_id]],
        #             "child_scores": [c["score"] for c in parent_children[parent_id]],
        #             "max_child_score": score
        #         }
        #     }
        # })
        parent_results.append({
            "id": parent_id,
            "text": parent_doc["text"],
            "score": score,
            "passage_id": parent_id,
            "chunk_type": "parent",
            "size": parent_doc.get("size", len(parent_doc["text"])),
            "num_matching_children": len(parent_children[parent_id]),
            "matching_children": parent_children[parent_id],

            "child_chunks": child_chunks,

            # Aggregated images
            "images": parent_images.get(parent_id, []),
            "image_anchors": parent_images.get(parent_id, []),  # Legacy compat

            "metadata": {
                **parent_doc.get("metadata", {}),
                "book_id": book_id,              # NEW
                "chapter_id": chapter_id,        # NEW
                "chunk_id": parent_id,           # NEW: stable citation id for parent
                "index_type": "detail",          # NEW (or keep if you want explicit)
                "expansion": {
                    "from_children": [c["child_id"] for c in parent_children[parent_id]],
                    "child_scores": [c["score"] for c in parent_children[parent_id]],
                    "max_child_score": score,
                },
            },
        })

    
    logger.info(f"[EXPAND] ✓ Expanded {len(child_results)} children to {len(parent_results)} unique parents")
    
    return parent_results


def get_parent_document(
    parent_id: str,
    book_id: str,
    chapter_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get a specific parent document by ID.
    
    Use cases:
    - Retrieving parent context after reflection
    - Debugging parent-child relationships
    - Direct parent access without search
    - Displaying full context for a passage
    
    Args:
        parent_id: Parent document ID (e.g., "parent_5")
        book_id: Book identifier
        chapter_id: Chapter identifier
    
    Returns:
        Parent document dict or None if not found
    """
    logger.debug(f"[GET PARENT] Fetching {parent_id} from {book_id}/{chapter_id}")
    
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    parents_path = shard_dir / "parents.jsonl"
    
    if not parents_path.exists():
        logger.warning(f"[GET PARENT] Parents file not found: {parents_path}")
        return None
    
    with open(parents_path, 'r', encoding='utf-8') as f:
        for line in f:
            parent = json.loads(line)
            if parent["id"] == parent_id:
                logger.debug(f"[GET PARENT] ✓ Found {parent_id}: {len(parent['text'])} chars")
                return parent
    
    logger.warning(f"[GET PARENT] Parent {parent_id} not found")
    return None


def get_child_chunks_for_parent(
    parent_id: str,
    book_id: str,
    chapter_id: str
) -> List[Dict[str, Any]]:
    """
    Get all child chunks belonging to a specific parent document.
    
    Use cases:
    - Debugging parent-child relationships
    - Viewing how a parent was chunked
    - Understanding child overlap strategy
    - Analyzing retrieval granularity
    
    Args:
        parent_id: Parent document ID (e.g., "parent_5")
        book_id: Book identifier
        chapter_id: Chapter identifier
    
    Returns:
        List of child chunk dicts belonging to this parent (sorted by local index)
    """
    logger.debug(f"[GET CHILDREN] Finding children for {parent_id} in {book_id}/{chapter_id}")
    
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    
    # Load parent-child mapping
    mapping_path = shard_dir / "parent_map.jsonl"
    if not mapping_path.exists():
        logger.warning(f"[GET CHILDREN] Mapping not found: {mapping_path}")
        return []
    
    # Find child IDs for this parent
    child_ids = []
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            mapping = json.loads(line)
            if mapping["parent_id"] == parent_id:
                child_ids.append(mapping["child_id"])
    
    if not child_ids:
        logger.warning(f"[GET CHILDREN] No children found for parent {parent_id}")
        return []
    
    logger.debug(f"[GET CHILDREN] Found {len(child_ids)} child IDs for {parent_id}")
    
    # Load child chunks from detail index
    chunks_path = shard_dir / "detail_chunks.jsonl"
    if not chunks_path.exists():
        logger.warning(f"[GET CHILDREN] Chunks file not found: {chunks_path}")
        return []
    
    children = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["id"] in child_ids:
                children.append(chunk)
    
    # Sort by local index (original order within parent)
    children.sort(key=lambda x: x.get("metadata", {}).get("local_index", 0))
    
    logger.info(f"[GET CHILDREN] ✓ Found {len(children)} children for parent {parent_id}")
    
    return children


def retrieve_with_hybrid_strategy(
    query: str,
    book_id: str,
    chapter_id: str,
    top_k_per_index: int = 3
) -> Dict[str, Any]:
    """
    Hybrid retrieval: Query both detail and summary indices, merge results.
    
    Use cases:
    - Questions benefiting from both detailed passages and high-level summaries
    - Example: "Explain photosynthesis and what topics this chapter covers"
    - Comprehensive answers requiring multiple perspectives
    
    Args:
        query: Search query
        book_id: Book identifier
        chapter_id: Chapter identifier
        top_k_per_index: Number of results from each index
    
    Returns:
        {
            "detail_results": [...],
            "summary_results": [...],
            "merged_results": [...]  # Interleaved and deduplicated
        }
    """
    logger.info(f"[HYBRID RETRIEVE] Query: '{query[:50]}...'")
    
    # Retrieve from detail index
    detail_results = retrieve_passages(
        query=query,
        book_id=book_id,
        chapter_id=chapter_id,
        index_type="detail",
        top_k=top_k_per_index,
        expand_to_parents=True
    )
    
    # Retrieve from summary index
    summary_results = retrieve_passages(
        query=query,
        book_id=book_id,
        chapter_id=chapter_id,
        index_type="summary",
        top_k=top_k_per_index,
        expand_to_parents=False
    )
    
    # Merge and sort by score
    merged = detail_results + summary_results
    merged.sort(key=lambda x: x["score"], reverse=True)
    
    # Deduplicate by ID
    seen_ids = set()
    deduplicated = []
    for result in merged:
        if result["id"] not in seen_ids:
            deduplicated.append(result)
            seen_ids.add(result["id"])
    
    logger.info(f"[HYBRID RETRIEVE] ✓ Merged {len(detail_results)} detail + "
                f"{len(summary_results)} summary → {len(deduplicated)} deduplicated")
    
    return {
        "detail_results": detail_results,
        "summary_results": summary_results,
        "merged_results": deduplicated[:top_k_per_index * 2]
    }


def get_summary_metadata(
    book_id: str,
    chapter_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve summary generation metadata.
    
    Metadata includes:
    - Provider and model used (e.g., "lmstudio/llama-3.2-3b-instruct")
    - Seed for reproducibility
    - Timestamp for version tracking
    - Input/output character counts
    - Compression ratio
    - Generation status (success/fallback)
    
    Use cases:
    - Auditing which provider generated summary
    - Checking if summary needs regeneration (timestamp)
    - Debugging summary quality issues (seed, input_chars)
    - Compliance and traceability
    
    Args:
        book_id: Book identifier
        chapter_id: Chapter identifier
    
    Returns:
        Metadata dict or None if not found
    """
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    metadata_path = shard_dir / "summary_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            logger.debug(f"[METADATA] Retrieved for {book_id}/{chapter_id}")
            return metadata
    
    logger.warning(f"[METADATA] Not found: {metadata_path}")
    return None


def get_retrieval_stats(
    book_id: str,
    chapter_id: str
) -> Dict[str, Any]:
    """
    Get retrieval statistics and health check for a chapter.
    
    Checks:
    - Shard directory existence
    - Detail index presence and chunk count
    - Summary index presence and chunk count
    - Parent documents and mapping files
    - Summary metadata availability
    
    Use cases:
    - Verifying ingestion completeness
    - Debugging retrieval issues
    - Monitoring system health
    
    Args:
        book_id: Book identifier
        chapter_id: Chapter identifier
    
    Returns:
        Stats dict with counts, sizes, and status
    """
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    
    stats = {
        "book_id": book_id,
        "chapter_id": chapter_id,
        "shard_exists": shard_dir.exists()
    }
    
    if not shard_dir.exists():
        stats["error"] = "Shard directory not found"
        return stats
    
    # Detail index stats
    stats["detail_index"] = {
        "exists": (shard_dir / "detail.index").exists(),
        "chunks_exist": (shard_dir / "detail_chunks.jsonl").exists(),
        "num_chunks": count_lines(shard_dir / "detail_chunks.jsonl"),
        "manifest_exists": (shard_dir / "detail_manifest.json").exists()
    }
    
    # Summary index stats
    stats["summary_index"] = {
        "exists": (shard_dir / "summary.index").exists(),
        "chunks_exist": (shard_dir / "summary_chunks.jsonl").exists(),
        "num_chunks": count_lines(shard_dir / "summary_chunks.jsonl"),
        "manifest_exists": (shard_dir / "summary_manifest.json").exists(),
        "summary_text_exists": (shard_dir / "chapter_summary.txt").exists()
    }
    
    # Parent-child structure stats
    stats["parent_child"] = {
        "parents_exist": (shard_dir / "parents.jsonl").exists(),
        "mapping_exists": (shard_dir / "parent_map.jsonl").exists(),
        "num_parents": count_lines(shard_dir / "parents.jsonl"),
        "num_mappings": count_lines(shard_dir / "parent_map.jsonl")
    }
    
    # Metadata stats
    stats["metadata"] = {
        "summary_metadata_exists": (shard_dir / "summary_metadata.json").exists()
    }
    
    logger.debug(f"[STATS] Retrieved for {book_id}/{chapter_id}")
    
    return stats


def validate_parent_child_consistency(
    book_id: str,
    chapter_id: str
) -> Dict[str, Any]:
    """
    Validate parent-child structure consistency.
    
    Checks:
    - All children have valid parent_id in chunks
    - All parent_ids in mapping exist in parents.jsonl
    - All children in mapping exist in detail_chunks.jsonl
    - No orphaned children (children without mapping)
    - Parent-child ID consistency
    
    Use cases:
    - Quality assurance after ingestion
    - Debugging retrieval issues
    - Validating re-ingestion
    
    Args:
        book_id: Book identifier
        chapter_id: Chapter identifier
    
    Returns:
        Validation report dict with errors, warnings, and statistics
    """
    logger.info(f"[VALIDATE] Checking parent-child consistency for {book_id}/{chapter_id}")
    
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    
    report = {
        "book_id": book_id,
        "chapter_id": chapter_id,
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required files exist
    parents_path = shard_dir / "parents.jsonl"
    mapping_path = shard_dir / "parent_map.jsonl"
    chunks_path = shard_dir / "detail_chunks.jsonl"
    
    if not parents_path.exists():
        report["valid"] = False
        report["errors"].append("parents.jsonl not found")
        return report
    
    if not mapping_path.exists():
        report["valid"] = False
        report["errors"].append("parent_map.jsonl not found")
        return report
    
    if not chunks_path.exists():
        report["valid"] = False
        report["errors"].append("detail_chunks.jsonl not found")
        return report
    
    # Load parent IDs
    parent_ids = set()
    with open(parents_path, 'r', encoding='utf-8') as f:
        for line in f:
            parent = json.loads(line)
            parent_ids.add(parent["id"])
    
    # Load child IDs and their parent_ids from chunks
    child_ids = set()
    child_parent_ids_from_chunks = {}
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            child_ids.add(chunk["id"])
            child_parent_ids_from_chunks[chunk["id"]] = chunk.get("parent_id")
    
    # Load and validate mappings
    mappings = []
    mapped_children = set()
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            mapping = json.loads(line)
            mappings.append(mapping)
            mapped_children.add(mapping["child_id"])
            
            # Validate child exists
            if mapping["child_id"] not in child_ids:
                report["valid"] = False
                report["errors"].append(f"Mapping references non-existent child: {mapping['child_id']}")
            
            # Validate parent exists
            if mapping["parent_id"] not in parent_ids:
                report["valid"] = False
                report["errors"].append(f"Mapping references non-existent parent: {mapping['parent_id']}")
            
            # Check consistency between mapping and chunk parent_id
            chunk_parent = child_parent_ids_from_chunks.get(mapping["child_id"])
            if chunk_parent and chunk_parent != mapping["parent_id"]:
                report["warnings"].append(
                    f"Child {mapping['child_id']}: parent_id mismatch "
                    f"(chunk={chunk_parent}, mapping={mapping['parent_id']})"
                )
    
    # Check for orphaned children
    orphaned_children = child_ids - mapped_children
    if orphaned_children:
        report["warnings"].append(
            f"Found {len(orphaned_children)} orphaned children without mappings"
        )
    
    # Add statistics
    report["stats"] = {
        "num_parents": len(parent_ids),
        "num_children": len(child_ids),
        "num_mappings": len(mappings),
        "num_orphaned": len(orphaned_children),
        "avg_children_per_parent": round(len(child_ids) / len(parent_ids), 2) if parent_ids else 0
    }
    
    if report["valid"]:
        logger.info(f"[VALIDATE] ✓ Structure is valid")
    else:
        logger.error(f"[VALIDATE] ✗ Found {len(report['errors'])} errors")
    
    if report["warnings"]:
        logger.warning(f"[VALIDATE] ⚠ Found {len(report['warnings'])} warnings")
    
    return report


def clear_cache_for_chapter(book_id: str, chapter_id: str):
    """
    Clear retrieval cache for specific chapter.
    
    Use after:
    - Re-ingesting a chapter
    - Updating parent-child mappings
    - Modifying indices
    - Debugging cache issues
    
    Args:
        book_id: Book identifier
        chapter_id: Chapter identifier
    """
    logger.info(f"[CACHE CLEAR] Clearing cache for {book_id}/{chapter_id}")
    _cache.clear(book_id, chapter_id)
    logger.info(f"[CACHE CLEAR] ✓ Cache cleared")


def count_lines(file_path: Path) -> int:
    """Count lines in a JSONL file"""
    if not file_path.exists():
        return 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

# -----------------------------------------------------------------------------
# NEW FUNCTIONALITY: Tier 4 (Subject-Wide) Retrieval
# -----------------------------------------------------------------------------

async def retrieve_all_chapter_summaries(
    book_id: str,
    top_k_per_chapter: int = 1
) -> List[Dict[str, Any]]:
    """
    Deterministically retrieve summaries for ALL chapters in a book using Async IO (aiofiles).
    Used for Tier 4 comprehensive/subject-wide queries (e.g., "question paper for entire subject").
    
    Args:
        book_id: The identifier of the book (e.g., "Class10Science").
        top_k_per_chapter: Number of summary chunks to take per chapter (default 1).
        
    Returns:
        List of result dictionaries containing summary text and metadata from every chapter.
    """
    results: List[Dict[str, Any]] = []
    shards_root = Path(settings.shards_dir)
    
    if not shards_root.exists():
        logger.warning(f"[SUMMARY ALL] Shards root not found: {shards_root}")
        return []

    # Find all relevant chapter directories
    # Convention: {book_id}_CH*
    pattern = f"{book_id}_*"
    # Glob is sync, but fast on directory listing.
    shard_dirs = sorted([d for d in shards_root.glob(pattern) if d.is_dir()])
    
    logger.info(f"[SUMMARY ALL] Scanning {len(shard_dirs)} chapters matching '{pattern}' in {shards_root}")

    async def process_chapter(shard_dir: Path) -> List[Dict[str, Any]]:
        """Helper to process a single chapter asynchronously."""
        chapter_results = []
        
        # Extract chapter_id from folder name
        folder_name = shard_dir.name
        # Expecting "Class10Science_CH1" -> "CH1"
        if folder_name.startswith(f"{book_id}_"):
            chapter_id = folder_name[len(book_id)+1:] 
        else:
            # Fallback if folder naming is unusual
            chapter_id = folder_name 

        summary_file = shard_dir / "summary_chunks.jsonl"
        if not summary_file.exists():
            # No summary index for this folder
            return []

        try:
            chunks_added = 0
            # Async file read with aiofiles
            async with aiofiles.open(summary_file, mode='r', encoding='utf-8') as f:
                async for line in f:
                    if chunks_added >= top_k_per_chapter:
                        break
                    
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Construct standardized result document
                    result_doc = {
                        "id": f"{chapter_id}_{chunk.get('id', 'summary')}",
                        "text": chunk.get("text", ""),
                        "score": 1.0, # Deterministic score for comprehensive inclusion
                        "metadata": {
                            **chunk.get("metadata", {}),
                            "book_id": book_id,
                            "chapter_id": chapter_id,
                            "retrieval_strategy": "summary_sampler_all"
                        }
                    }
                    chapter_results.append(result_doc)
                    chunks_added += 1
        except Exception as e:
            logger.error(f"[SUMMARY ALL] Error reading summaries for {chapter_id}: {e}")
            return []
            
        return chapter_results

    # Run all chapter reads concurrently
    # This utilizes the event loop to wait on I/O without blocking
    tasks = [process_chapter(d) for d in shard_dirs]
    if not tasks:
        return []

    chapter_batches = await asyncio.gather(*tasks)
    
    # Flatten list of lists
    for batch in chapter_batches:
        results.extend(batch)

    logger.info(f"[SUMMARY ALL] Retrieved {len(results)} summary chunks across all chapters for {book_id}")
    return results

from typing import Optional # Ensure this is imported

async def retrieve_book_structure(book_id: str, chapter_id: Optional[str] = None) -> str:
    """
    Retrieves structural metadata (TOC).
    
    - If chapter_id is provided: Returns TOC for that specific chapter.
    - If no chapter_id: Aggregates TOC from ALL chapters in the book.
    
    Returns:
        A Markdown-formatted string representing the hierarchy.
    """
    # FIX 1: Use correct settings attribute (was settings.SHARDS_ROOT)
    shard_root = Path(settings.shards_dir)
    
    if not shard_root.exists():
        return f"Error: Shard directory not found at {shard_root}"

    # FIX 2: Smart Targeting - scan only what is needed
    if chapter_id:
        # Targeted lookup: Specific chapter shard
        target_dir = shard_root / f"{book_id}_{chapter_id}"
        chapter_dirs = [target_dir] if target_dir.exists() else []
    else:
        # Aggregate lookup: All chapters for this book
        chapter_dirs = sorted([
            d for d in shard_root.iterdir() 
            if d.is_dir() and d.name.startswith(f"{book_id}_")
        ])

    if not chapter_dirs:
        target_msg = f"Chapter {chapter_id}" if chapter_id else f"Book {book_id}"
        return f"No structure found for {target_msg}"

    structure_lines = [f"# Structure of {book_id}"]
    if chapter_id:
        structure_lines.append(f"(Focus: {chapter_id})")

    for chapter_dir in chapter_dirs:
        toc_path = chapter_dir / "toc.json"
        chapter_name = chapter_dir.name.replace(f"{book_id}_", "")
        
        structure_lines.append(f"\n## Chapter: {chapter_name}")
        
        if toc_path.exists():
            try:
                with open(toc_path, 'r', encoding='utf-8') as f:
                    toc_data = json.load(f)
                # async with aiofiles.open(toc_path, mode='r', encoding='utf-8') as f:
                #     content = await f.read()
                #     toc_data = json.loads(content)
                    
                if not toc_data:
                    structure_lines.append("(No headings found)")
                    continue

                # Format extracted headings
                for item in toc_data:
                    # Indent based on level (2 spaces per level)
                    indent = "  " * (item.get('level', 1) - 1)
                    title = item.get('title', 'Untitled')
                    structure_lines.append(f"{indent}- {title}")
            except Exception as e:
                logger.error(f"Error reading TOC for {chapter_name}: {e}")
                structure_lines.append("(Error loading structure)")
        else:
            structure_lines.append("(Structure metadata missing)")

    return "\n".join(structure_lines)



# Export public API - Complete list for external use
__all__ = [
    # Core retrieval
    "retrieve_passages",
    "retrieve_passages_batch",
    
    # Parent-child operations
    "expand_to_parent_documents",
    "get_parent_document",
    "get_child_chunks_for_parent",
    
    # Hybrid and advanced
    "retrieve_with_hybrid_strategy",
    
    # Metadata and stats
    "get_summary_metadata",
    "get_retrieval_stats",
    
    # Validation and maintenance
    "validate_parent_child_consistency",
    "clear_cache_for_chapter",
    
    # New Tier 4 retrieval function
    "retrieve_all_chapter_summaries" 
    
    # Increment 11.2: Structure Retrieval
    "retrieve_book_structure"
]

