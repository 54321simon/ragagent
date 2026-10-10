"""Compatibility entry point for the upgraded multi-document benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.evaluate import run

if __name__ == "__main__":
    run(["vector", "hybrid", "rerank"])
