"""Bounded Ollama HTTP calls, shared by all application modules."""

import os
import ollama as sdk
from common.config import MODEL


def client():
    return sdk.Client(
        host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        timeout=float(os.getenv("LLM_TIMEOUT", "180")),
        trust_env=False,
    )


def chat(**kwargs):
    kwargs.setdefault("model", MODEL)
    options = {
        "num_ctx": int(os.getenv("MODEL_CONTEXT", "4096")),
        "num_predict": int(os.getenv("MAX_OUTPUT_TOKENS", "512")),
    }
    options.update(kwargs.get("options") or {})
    kwargs["options"] = options
    return client().chat(**kwargs)


def embed(**kwargs):
    return client().embed(**kwargs)


def list():
    return client().list()
