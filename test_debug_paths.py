from pathlib import Path
from backend.config import get_settings
import json

settings = get_settings()
root = Path(settings.shards_dir)

print(f"Shards Root: {root.absolute()}")

# Expected Path
target_dir = root / "Class10Science_CH1"
target_file = target_dir / "toc.json"

print(f"Target Dir Exists? {target_dir.exists()}")
if target_dir.exists():
    print(f"Contents of dir: {[f.name for f in target_dir.iterdir()]}")

print(f"TOC File Exists? {target_file.exists()}")

if target_file.exists():
    with open(target_file, 'r') as f:
        content = f.read()
        print(f"TOC Content Length: {len(content)}")
        print(f"TOC Content Preview: {content[:100]}")
