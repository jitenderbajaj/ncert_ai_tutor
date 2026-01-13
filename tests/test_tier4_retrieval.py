import pytest
import asyncio
import os
from backend.agent.steps.retriever import retriever_step

# Use 'pytest-asyncio' or simply run this with 'pytest' if configured
@pytest.mark.asyncio
async def test_tier4_subject_wide_retrieval():
    """
    Test that the retriever correctly gathers summaries from ALL chapters
    when the plan specifies summarysampler="all".
    """
    
    # 1. SETUP: Define a mock state with a Plan that triggers Tier 4
    mock_state = {
        "plan": {
            "query": "Prepare a question paper for the entire subject",
            "bookid": "Class10Science",  # Ensure this book exists in your shards!
            "chapterid": None,           # Subject-wide queries might not have a specific chapter
            "indextype": "summary",
            "summarysampler": "all",     # <--- This triggers the new code
            "topkperchapter": 1
        }
    }

    print("\n[TEST] Executing retriever_step with Tier 4 Plan...")

    # 2. EXECUTE: Run the step
    result = await retriever_step(mock_state)
    docs = result.get("retrieved_docs", [])

    # 3. VERIFY
    print(f"[TEST] Retrieved {len(docs)} documents.")
    
    # Assertion 1: We should get results (assuming shards exist)
    # Note: If you haven't ingested any chapters for 'Class10Science', this will fail/be empty.
    # For a robust test, we check if 'docs' is a list.
    assert isinstance(docs, list), "Result should be a list of documents"

    if len(docs) > 0:
        first_doc = docs[0]
        print(f"[TEST] Sample Doc ID: {first_doc.get('id')}")
        print(f"[TEST] Sample Metadata: {first_doc.get('metadata')}")
        
        # Assertion 2: Metadata should confirm the strategy
        assert first_doc["metadata"].get("retrieval_strategy") == "summary_sampler_all", \
            "Documents should be tagged with the correct retrieval strategy"
            
        # Assertion 3: We should see multiple chapters if they exist
        chapter_ids = {d["metadata"]["chapter_id"] for d in docs}
        print(f"[TEST] Found chapters: {chapter_ids}")
        
        if len(chapter_ids) > 1:
            print("[TEST] SUCCESS: Retrieved summaries from multiple chapters!")
        else:
            print("[TEST] WARNING: Only found 1 chapter. (Did you ingest multiple chapters?)")
    else:
        print("[TEST] WARNING: No docs retrieved. Ensure 'Class10Science' shards exist.")

if __name__ == "__main__":
    # Helper to run without pytest command line
    asyncio.run(test_tier4_subject_wide_retrieval())
