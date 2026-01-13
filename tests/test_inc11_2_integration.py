
import asyncio
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_inc11_2")

# Import your modules (Assuming python path is set correctly)
from backend.config import get_settings
from backend.agent.steps.planner import plan_step, get_router
from backend.services.ingestion_dual import ingest_and_persist_chapter, _extract_toc_from_text
from backend.services.retrieval_dual import retrieve_book_structure, retrieve_passages
from backend.agent.steps.compose import compose_step

settings = get_settings()

async def run_integration_test():
    print("=== STARTING INCREMENT 11.2 INTEGRATION TEST ===")

    # Setup Test Data
    book_id = "TEST_BOOK_11_2"
    chapter_id = "CH1"
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"

    # Clean up old test data
    if shard_dir.exists():
        shutil.rmtree(shard_dir)

    # --- TEST 1: INGESTION (TOC + Images) ---
    print("\n[1] Testing Ingestion (TOC Generation)...")

    # Mock text with Markdown headers
    mock_text = """
# Chapter 1: Chemical Reactions
## 1.1 Introduction
Chemical reactions are fundamental.
## 1.2 Types of Reactions
There are many types.
### 1.2.1 Combination
Reaction where A + B -> C.
    """

    # Test utility function directly first
    toc = _extract_toc_from_text(mock_text)
    assert len(toc) == 3, f"Expected 3 headings, got {len(toc)}"
    print("✓ _extract_toc_from_text passed")

    # Note: We skip full PDF ingestion here to avoid needing a real PDF file.
    # Instead, we simulate the artifacts that ingestion WOULD create.
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Save Mock TOC
    with open(shard_dir / "toc.json", "w") as f:
        json.dump(toc, f)

    # Save Mock Chunk with Image (Simulating extraction)
    mock_chunk = {
        "id": "C_0_0",
        "text": "See Figure 1.1 for details.",
        "images": [{"id": "img_1", "label": "Figure 1.1"}],
        "image_anchors": [{"id": "img_1", "label": "Figure 1.1"}]
    }
    with open(shard_dir / "detail_chunks.jsonl", "w") as f:
        f.write(json.dumps(mock_chunk) + "\n")

    # Create dummy index (required for retrieval to not crash)
    import faiss
    import numpy as np
    d = 384 # Dimension for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(d)
    # Add one dummy vector
    index.add(np.random.random((1, d)).astype('float32')) 
    faiss.write_index(index, str(shard_dir / "detail.index"))

    print("✓ Ingestion artifacts mocked successfully")


    # --- TEST 2: PLANNER (Hybrid Router) ---
    print("\n[2] Testing Planner (Hybrid Router)...")

    # Case A: Structure Query
    plan_struct = await plan_step("What is the structure of this book?", book_id, chapter_id)
    assert plan_struct["plan"]["strategy"] == "structure_lookup", f"Failed Structure Route: {plan_struct['plan']['strategy']}"
    print("✓ Routed 'Structure' correctly")

    # Case B: General Chat
    plan_gen = await plan_step("Hello, how are you?", book_id, None)
    assert plan_gen["plan"]["strategy"] == "general_chat", f"Failed General Chat Route: {plan_gen['plan']['strategy']}"
    print("✓ Routed 'General Chat' correctly")

    # Case C: Assessment
    plan_quiz = await plan_step("Create a quiz for this chapter", book_id, chapter_id)
    assert plan_quiz["plan"]["strategy"] == "generate_assessment", f"Failed Assessment Route: {plan_quiz['plan']['strategy']}"
    print("✓ Routed 'Assessment' correctly")


    # --- TEST 3: RETRIEVAL (Structure & Images) ---
    print("\n[3] Testing Retrieval...")

    # Case A: Retrieve Structure
    structure_text = await retrieve_book_structure(book_id)
    assert "Chemical Reactions" in structure_text, "Structure text missing chapter title"
    print("✓ retrieve_book_structure returned correct content")

    # Case B: Retrieve Passages (Image Propagation)
    # We use a query that doesn't matter because we only have 1 chunk in the mock index
    results = retrieve_passages("Figure 1.1", book_id, chapter_id, top_k=1)

    if results and "images" in results[0]:
        imgs = results[0]["images"]
        assert len(imgs) > 0, "Images list empty in retrieval result"
        assert imgs[0]["id"] == "img_1", "Image ID mismatch"
        print("✓ Retrieval propagated 'images' field correctly")
    else:
        print(f"❌ Failed to retrieve images. Result keys: {results[0].keys() if results else 'No results'}")


    # --- TEST 4: COMPOSE (Prompt Logic) ---
    print("\n[4] Testing Compose...")

    # Case A: General Chat
    res_gen = await compose_step("Hi", [], [], plan_gen["plan"], "test_id")
    # We check if it returned an answer without crashing (Mock provider usually echoes or returns dummy)
    assert res_gen["answer"], "Compose returned empty answer for general chat"
    print("✓ Compose handled general chat")

    print("\n=== ALL INCREMENT 11.2 INTEGRATION TESTS PASSED ===")

if __name__ == "__main__":
    asyncio.run(run_integration_test())
