"""Batch import with per-document status and bounded retries."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.api import ingest_with_retry


def main():
    records = []
    for path in sorted((ROOT / "data/papers").glob("*.pdf")):
        try:
            count = ingest_with_retry(str(path))
            records.append({"doc_id": path.stem, "chunks": count, "success": True})
        except Exception as exc:
            records.append({"doc_id": path.stem, "success": False, "error": str(exc)})
        print(records[-1], flush=True)
    (ROOT / "experiments/results/ingestion.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if any(not r["success"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
