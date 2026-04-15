import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

_persist_dir = os.getenv("INDEX_PERSIST_DIR")
if _persist_dir:
    PERSIST_DIR = Path(_persist_dir)
else:
    PERSIST_DIR = BASE_DIR / "storage" / "index_v2"


