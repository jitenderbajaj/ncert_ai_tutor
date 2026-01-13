import os
from pathlib import Path

# Update this path to your actual project root
ROOT_DIR = Path("D:/Jitender/AI/Code/Capstone/ncert_ai_tutor_i11.01")
SHARDS_DIR = ROOT_DIR / "data" / "shards"

print(f"Checking shards in: {SHARDS_DIR}")

if not SHARDS_DIR.exists():
    print("❌ Shards directory NOT found!")
else:
    print("✅ Shards directory exists.")
    shards = list(SHARDS_DIR.glob("*"))
    print(f"Found {len(shards)} shards:")
    for s in shards:
        print(f" - {s.name}")
        if (s / "detail.index").exists():
            print("   ✅ detail.index found")
        else:
            print("   ❌ detail.index MISSING")
