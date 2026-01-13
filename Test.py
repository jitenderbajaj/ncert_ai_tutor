# FILE: test.py (FIXED VERSION)

# Test complete pipeline
from backend.services.ingestion_dual import ingest_chapter_dual

result = ingest_chapter_dual(
    pdf_path=r"D:\Jitender\AI\Code\Capstone\ncert_ai_tutor_i11\datasource\class10\science\CH1.pdf",
    book_id="Class10Science",
    chapter_id="CH1",
    seed=42
)

# Check results
print(f"Status: {result['status']}")
print(f"Images extracted: {len(result['images'])}")
print(f"Detail chunks: {result['parent_child_stats']['num_children']}")
print(f"Summary chunks: {result['summary_metadata']['output_chars']} chars")

# Check if images are bound to chunks
import json
from pathlib import Path

shard_dir = Path("data/shards/Class10Science_CH1")

# FIX: Add encoding='utf-8'
with open(shard_dir / "detail_chunks.jsonl", 'r', encoding='utf-8') as f:
    chunks_with_images = 0
    total_chunks = 0
    for line in f:
        chunk = json.loads(line)
        total_chunks += 1
        if chunk.get("image_anchors"):
            chunks_with_images += 1
            print(f"Chunk {chunk['id']} has {len(chunk['image_anchors'])} images")

print(f"\nChunks with images: {chunks_with_images}/{total_chunks}")
