"""Append structured request records without mixing model score types."""

import json
import threading
from datetime import datetime
from common.config import LOG_DIR

_LOCK = threading.Lock()


def write_log(kind, record):
    record = {"ts": datetime.now().astimezone().isoformat(), **record}
    with _LOCK:
        with (LOG_DIR / f"{kind}_{datetime.now():%Y%m%d}.jsonl").open(
            "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
