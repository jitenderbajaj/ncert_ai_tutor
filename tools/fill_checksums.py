# FILE: tools/fill_checksums.py
"""
Fill checksums in MANIFEST.json (staged mode)
"""
import json
import hashlib
from pathlib import Path

def compute_sha256(file_path):
    """Compute SHA256 checksum of file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def fill_checksums(manifest_path="MANIFEST.json", root_dir="."):
    """Fill checksums in MANIFEST.json"""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    for file_entry in manifest["files"]:
        path = Path(root_dir) / file_entry["path"]
        if path.exists():
            checksum = compute_sha256(path)
            file_entry["sha256"] = checksum
            print(f"✓ {file_entry['path']}: {checksum[:8]}...")
        else:
            print(f"✗ {file_entry['path']}: NOT FOUND")
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nChecksums filled in {manifest_path}")

if __name__ == "__main__":
    fill_checksums()
