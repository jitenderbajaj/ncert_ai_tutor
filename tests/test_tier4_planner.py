import pytest
import asyncio
from backend.agent.steps.planner import plan_step

@pytest.mark.asyncio
async def test_tier4_planner_intent():
    """
    Test that the Planner correctly identifies a subject-wide assessment intent
    and sets summarysampler="all".
    """
    
    # Define inputs matching the function signature
    query = "Prepare a question paper for the entire subject"
    book_id = "Class10Science"
    
    print(f"\n[TEST] Executing plan_step('{query}', '{book_id}')...")

    # CALL THE FUNCTION DIRECTLY with correct args
    # Note: plan_step in your file is NOT async, so we don't await it!
    # If it IS async in your local version, keep 'await'. 
    # Based on the file provided, it looks like a synchronous def.
    
    try:
        # Try sync first (based on attached file content 'def plan_step...')
        plan = plan_step(question=query, book_id=book_id)
    except TypeError:
        # Fallback if it IS async
        plan = await plan_step(question=query, book_id=book_id)
    
    # 3. INSPECT OUTPUT
    # The result IS the plan dict, not {"plan": ...}
    print(f"[TEST] Generated Plan Strategy: {plan.get('strategy')}")
    print(f"[TEST] Generated Summary Sampler: {plan.get('summary_sampler')}") # Note: check snake_case vs one word
    print(f"[TEST] Generated Index Hint: {plan.get('index_hint')}")

    # 4. VERIFY
    assert plan.get("strategy") == "generate_assessment", \
        f"Expected strategy 'generate_assessment', got '{plan.get('strategy')}'"
        
    # Your file uses 'summary_sampler', earlier code used 'summarysampler'. 
    # We check both to be safe, but the attached file implies snake_case keys.
    sampler = plan.get("summary_sampler") or plan.get("summarysampler")
    
    assert sampler == "all", \
        f"Expected summary_sampler='all', got '{sampler}'"

    print("[TEST] SUCCESS: Planner correctly configured a Tier 4 Subject-Wide plan!")

if __name__ == "__main__":
    asyncio.run(test_tier4_planner_intent())


