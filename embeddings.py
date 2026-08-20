"""Embedding backend.

Prefers sentence-transformers (a real local open model). If it is not installed,
falls back to a deterministic hashing bag-of-words embedding so retrieval still
functions offline. The fallback is weaker but keeps the whole system runnable
and testable without downloading a model.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from . import config

_ST_MODEL = None
_BACKEND = None


def _try_load_sentence_transformers():
    global _ST_MODEL, _BACKEND
    if _BACKEND is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _ST_MODEL = SentenceTransformer(config.EMBED_MODEL)
        _BACKEND = "sentence-transformers"
    except Exception:
        _ST_MODEL = None
        _BACKEND = "hashing"


_TOKEN = re.compile(r"[a-z0-9]+")
_HASH_DIM = 512


def _hash_embed(text: str) -> List[float]:
    vec = [0.0] * _HASH_DIM
    for tok in _TOKEN.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _HASH_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def backend() -> str:
    _try_load_sentence_transformers()
    return _BACKEND


def embed(texts: List[str]) -> List[List[float]]:
    _try_load_sentence_transformers()
    if _BACKEND == "sentence-transformers":
        return [list(map(float, v)) for v in _ST_MODEL.encode(texts, normalize_embeddings=True)]
    return [_hash_embed(t) for t in texts]


def embed_one(text: str) -> List[float]:
    return embed([text])[0]
