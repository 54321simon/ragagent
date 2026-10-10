"""Persistent semantic cache with corpus, scope, history and model isolation."""

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from common.config import DATA_DIR


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def protected_terms(query):
    # A numerically different question must never reuse a semantically close answer.
    return tuple(
        sorted(re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+[A-Za-z0-9_.-]*", query.lower()))
    )


class SemanticCache:
    def __init__(self, path=None, threshold=None, ttl=86400):
        self.path = str(path or DATA_DIR / "semantic_cache.sqlite3")
        self.threshold = float(threshold or os.getenv("CACHE_THRESHOLD", "0.97"))
        self.ttl = ttl
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cache (id TEXT PRIMARY KEY, scope TEXT, query TEXT, vector TEXT, payload TEXT, created REAL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS cache_scope ON cache(scope)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, query, scope, vector=None):
        key = fingerprint([query, scope])
        with self.connect() as db:
            db.execute("DELETE FROM cache WHERE created<?", (time.time() - self.ttl,))
            row = db.execute("SELECT payload FROM cache WHERE id=?", (key,)).fetchone()
            if row:
                return json.loads(row[0]), 1.0
            if vector is None:
                return None, 0.0
            rows = db.execute(
                "SELECT query,vector,payload FROM cache WHERE scope=? ORDER BY created DESC LIMIT 128",
                (scope,),
            ).fetchall()
        best = (None, 0.0)
        for old, v, payload in rows:
            if protected_terms(old) != protected_terms(query):
                continue
            v = json.loads(v)
            if len(v) != len(vector):
                continue
            den = math.sqrt(sum(x * x for x in v) * sum(x * x for x in vector)) or 1
            score = sum(x * y for x, y in zip(v, vector)) / den
            if score >= self.threshold and score > best[1]:
                best = (json.loads(payload), score)
        return best

    def put(self, query, scope, vector, payload):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
                (
                    fingerprint([query, scope]),
                    scope,
                    query,
                    json.dumps(vector),
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
            db.execute(
                "DELETE FROM cache WHERE id NOT IN (SELECT id FROM cache ORDER BY created DESC LIMIT 1000)"
            )

    def clear(self):
        with self.connect() as db:
            db.execute("DELETE FROM cache")
