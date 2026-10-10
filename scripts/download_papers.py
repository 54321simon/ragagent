"""Download the public paper corpus and verify recorded PDF hashes."""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import truststore

    truststore.inject_into_ssl()
    target = ROOT / "data/papers"
    target.mkdir(parents=True, exist_ok=True)
    for item in json.loads(
        (ROOT / "experiments/corpus_manifest.json").read_text(encoding="utf-8")
    ):
        path = target / (item["doc_id"] + ".pdf")
        if not path.exists():
            request = urllib.request.Request(
                item["pdf_url"], headers={"User-Agent": "NJAU-Coursework/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            if not content.startswith(b"%PDF"):
                raise ValueError("The download is not a PDF: " + item["doc_id"])
            path.write_bytes(content)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            print(
                "WARNING: upstream PDF version differs; review physical page labels:",
                item["doc_id"],
                file=sys.stderr,
            )
        print(item["doc_id"], actual, flush=True)


if __name__ == "__main__":
    main()
