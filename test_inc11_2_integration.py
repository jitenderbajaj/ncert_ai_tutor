import asyncio
import logging
import json
import shutil
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

# --- UPDATE: Enable DEBUG logging to see Prompts/Responses ---
logging.basicConfig(
    level=logging.DEBUG, # Changed to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Filter out noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger("test_inc11_2")

# Import your modules
from backend.config import get_settings
from backend.agent.steps.planner import plan_step, get_router
from backend.services.retrieval_dual import retrieve_book_structure, retrieve_passages
from backend.agent.steps.compose import compose_step

settings = get_settings()

def _mock_extract_toc(text: str):
    toc = []
    lines = text.split('\n')
    md_pattern = re.compile(r'^(#{1,4})\s+(.*)')
    for line in lines:
        match = md_pattern.match(line.strip())
        if match:
            toc.append({
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "source": "markdown"
            })
    return toc

async def run_integration_test():
    print("\n=== STARTING INCREMENT 11.2 INTEGRATION TEST (DEBUG MODE) ===\n")
    
    # Setup Test Data
    book_id = "TEST_BOOK_11_2"
    chapter_id = "CH1"
    shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
    
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    
    # --- TEST 1: INGESTION ---
    print("\n[1] Testing Ingestion (Mock)...")
    mock_text = """
# Chapter 1: Chemical Reactions
## 1.1 Introduction
"""
    toc = _mock_extract_toc(mock_text)
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(shard_dir / "toc.json", "w") as f:
        json.dump(toc, f)
        
    mock_chunk = {
        "id": "C_0_0",
        "text": "See Figure 1.1 for details.",
        "images": [{"id": "img_1", "label": "Figure 1.1"}],
        "metadata": {}
    }
    with open(shard_dir / "detail_chunks.jsonl", "w") as f:
        f.write(json.dumps(mock_chunk) + "\n")
        
    import faiss
    import numpy as np
    d = 384 
    index = faiss.IndexFlatL2(d)
    index.add(np.random.random((1, d)).astype('float32')) 
    faiss.write_index(index, str(shard_dir / "detail.index"))
    print("✓ Ingestion artifacts mocked")

    # --- TEST 2: PLANNER ---
    print("\n[2] Testing Planner (Hybrid Router)...")
    
    # Case A
    print("\n--- Query A: Structure ---")
    q1 = "What is the structure of this book?"
    plan_struct = await plan_step(q1, book_id, chapter_id)
    print(f"DEBUG: Plan Result: {json.dumps(plan_struct, indent=2)}")
    
    # Case B (The one failing)
    print("\n--- Query B: General Chat ---")
    q2 = "Hello, how are you?"
    plan_gen = await plan_step(q2, book_id, None)
    print(f"DEBUG: Plan Result: {json.dumps(plan_gen, indent=2)}")
    
    # Assertion with detailed error if fail
    if plan_gen["plan"]["strategy"] != "general_chat":
        print(f"❌ FAIL: Expected 'general_chat', got '{plan_gen['plan']['strategy']}'")
        # We proceed to see retrieval logic anyway
    else:
        print("✓ Routed 'General Chat' correctly")

    # --- TEST 3: RETRIEVAL ---
    print("\n[3] Testing Retrieval...")
    results = retrieve_passages("Figure 1.1", book_id, chapter_id, top_k=1)
    print(f"DEBUG: Retrieved Chunk: {results[0] if results else 'None'}")

    # --- TEST 4: COMPOSE ---
    print("\n[4] Testing Compose...")
    # Test Compose with the General Chat plan
    print("--- Composing Response for General Chat ---")
    res_gen = await compose_step("Hi", [], [], plan_gen["plan"], "test_id")
    print(f"DEBUG: LLM Response: {res_gen['answer']}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(run_integration_test())
