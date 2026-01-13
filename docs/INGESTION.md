# FILE: docs/INGESTION.md
# Ingestion Guide — Multi-Index / Hybrid RAG Strategy

## Overview

The NCERT AI Tutor implements a **true Multi-Index / Hybrid RAG strategy** with two distinct knowledge sources:

### 1. Detail Index (Fine-Grained)
- **Content**: Small chunks (512 characters) with 50-character overlap
- **Purpose**: Provide specific, granular textual evidence for grounding answers
- **Use Case**: "Explain photosynthesis in detail", "What is the formula for...?"
- **Parent Document Retrieval**: Chunks include stable `passage_id` for parent mapping

### 2. Summary Index (Coarse-Grained)
- **Content**: LLM-generated chapter summaries (~500-1000 tokens per chapter)
- **Generation Method**: During ingestion, an LLM generates a concise, high-level summary of the entire chapter
- **Purpose**: Provide high-level context and topic coverage
- **Use Case**: "What topics does Chapter 5 cover?", "Summarize this chapter", "What are the main concepts?"

## Ingestion Pipeline

### Step 1: PDF Text Extraction
text = extract_text_from_pdf(pdf_path)

### Step 2: Detail Index Creation
Fine-grained chunks for specific details
detail_chunks = chunk_text(text, chunk_size=512, overlap=50, seed=42)
detail_manifest = build_index(detail_chunks, index_type="detail")

### Step 3: LLM Summary Generation
LLM generates high-level chapter summary
chapter_summary = generate_chapter_summary_with_llm(
chapter_text=text,
book_id=book_id,
chapter_id=chapter_id,
seed=42
)

**LLM Prompt Template:**
You are an expert NCERT textbook summarizer. Generate a concise, high-level summary of this chapter.

REQUIREMENTS:

Length: 500-1000 tokens

Focus on: main topics, key concepts, learning objectives, chapter structure

Style: Clear, educational, suitable for students

Format: Structured paragraphs covering each major section

Chapter Text:
[CHAPTER_TEXT]

High-Level Chapter Summary:


### Step 4: Summary Index Creation
Create chunks from LLM summary (usually 1 chunk per chapter)
summary_chunks = create_summary_chunks(chapter_summary)
summary_manifest = build_index(summary_chunks, index_type="summary")


### Step 5: Image Extraction
Extract figures and diagrams
images = extract_images_from_pdf(pdf_path, book_id, chapter_id)


## Planner Logic

The **planner agent** selects the appropriate index based on query intent:

if any(kw in question.lower() for kw in ["overview", "summarize", "main topics", "introduction", "what topics", "what concepts"]):
chosen_index = "summary" # Query coarse-grained LLM summary
else:
chosen_index = "detail" # Query fine-grained chunks

## Retrieval Flow

### Detail Index Query
1. Embed user question
2. Search detail index (fine-grained chunks)
3. Retrieve top-k passages with high similarity
4. Use for grounding specific answers

### Summary Index Query
1. Embed user question
2. Search summary index (LLM-generated summaries)
3. Retrieve chapter-level summaries
4. Use for high-level overviews and topic coverage

## Benefits

1. **Dual Granularity**: Fine details + high-level context
2. **Efficient Overview Queries**: Direct access to chapter summaries without scanning all chunks
3. **Better Grounding**: Specific chunks for detailed questions, summaries for overview questions
4. **LLM-Enhanced**: Summaries are coherent, well-structured, curriculum-aligned

## API Usage

### Ingest with LLM Summary
curl -X POST "http://localhost:8000/ingest/pdf"
-F "file=@Chapter1.pdf"
-F "book_id=BOOK123"
-F "chapter_id=CH1"
-F "seed=42"

**Response includes:**
{
"detail_manifest": {...},
"summary_manifest": {...},
"images": [...],
"summary_text": "# Chapter Summary: CH1\n\nThis chapter covers..."
}


### Query Detail Index
curl -X POST "http://localhost:8000/agent/answer"
-H "Content-Type: application/json"
-d '{
"question": "Explain the process of photosynthesis in detail",
"book_id": "BOOK123",
"chapter_id": "CH1",
"index_hint": "detail"
}'


### Query Summary Index
curl -X POST "http://localhost:8000/agent/answer"
-H "Content-Type: application/json"
-d '{
"question": "What are the main topics covered in Chapter 1?",
"book_id": "BOOK123",
"chapter_id": "CH1",
"index_hint": "summary"
}'


## Determinism

- **Seed**: Fixed seed (default 42) ensures reproducible LLM summaries
- **Temperature**: 0.0 for deterministic LLM output
- **Chunking**: Deterministic sentence splitting with stable passage IDs
- **Embeddings**: Fixed embedding model (all-MiniLM-L6-v2)
- **FAISS**: IndexFlatL2 with seeded index construction

## Fallback Behavior

If LLM is unavailable during ingestion:
- Falls back to first 2000 characters of chapter as "summary"
- Logs warning and continues ingestion
- Manifest includes `summary_generation: "fallback"`

## Validation

Check that summary index uses LLM-generated summaries:
cat data/shards/BOOK123_CH1/chapter_summary.txt
cat data/shards/BOOK123_CH1/summary_manifest.json


Look for:
- `"summary_generation": "llm"`
- `"chunk_type": "llm_generated_summary"`
- Summary text starting with metadata header

## Testing

def test_llm_summary_generation():
"""Test that summary index uses LLM-generated summaries"""
result = ingest_chapter_dual(
pdf_path="test.pdf",
book_id="TEST",
chapter_id="CH1",
seed=42
)
# Check summary manifest
assert result["summary_manifest"]["params"]["summary_generation"] == "llm"
assert "summary_text" in result
assert len(result["summary_text"]) > 0

# Check chunk type
with open("data/shards/TEST_CH1/summary_chunks.jsonl") as f:
    chunk = json.loads(f.readline())
    assert chunk["type"] == "llm_generated_summary"

---

**End of INGESTION.md**
