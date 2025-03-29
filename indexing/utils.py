import hashlib
from pathlib import Path

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of file"""
    print(f"Calculating hash for {file_path}")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()