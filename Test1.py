# Verify images are accessible
import json
from pathlib import Path

shard_dir = Path(r"D:\Jitender\AI\Code\Capstone\ncert_ai_tutor_i11\data\shards\Class10Science_CH1")

# Check one chunk with images
with open(shard_dir / "detail_chunks.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        chunk = json.loads(line)
        if chunk.get("image_anchors"):
            print(f"\n{chunk['id']}:")
            print(f"Text: {chunk['text'][:100]}...")
            print(f"Images:")
            for img in chunk['image_anchors']:
                # >>> MODIFICATION HERE: Print the dictionary to see its contents
                print(f"  - Dictionary Keys: {img.keys()}")
                print(f"  - Full Dictionary: {img}")
                
                # Use the correct key found from the output above (e.g., 'filepath')
                # img_path = Path(img['filepath']) # Placeholder for the fix
                
                # TEMPORARILY COMMENTING OUT THE FAULTY LINE TO RUN THE INSPECTION
                # img_path = Path(img['path']) 
                
                # You can stop here after printing the first image's keys
                break
            break