"""Load project configuration before constructing model clients."""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(DATA_DIR / "index"))).resolve()
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", str(DATA_DIR / "papers"))).resolve()
LOG_DIR = Path(os.getenv("LOG_DIR", str(DATA_DIR / "logs"))).resolve()
for directory in (DATA_DIR, INDEX_DIR, PAPERS_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
