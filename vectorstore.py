"""A tiny, dependency-free vector store.

Persists chunks + embeddings to a JSON file and does exact cosine-similarity
search in NumPy. For a corpus of clinical documents this is more than fast
enough, and it keeps the project runnable with zero external services. Swapping
in FAISS/Chroma/pgvector later is a drop-in change behind this interface.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    chunk_index: int


class VectorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.chunks: List[Chunk] = []
        self._matrix: np.ndarray | None = None

    # --- persistence ---------------------------------------------------------
    def save(self, embeddings: List[List[float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [asdict(c) for c in self.chunks],
            "embeddings": embeddings,
        }
        self.path.write_text(json.dumps(payload))

    def load(self) -> None:
        data = json.loads(self.path.read_text())
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self._matrix = np.asarray(data["embeddings"], dtype=np.float32)

    def exists(self) -> bool:
        return self.path.exists()

    # --- search --------------------------------------------------------------
    def search(self, query_embedding: List[float], top_k: int) -> List[tuple[Chunk, float]]:
        if self._matrix is None:
            self.load()
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        mat = self._matrix
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        sims = (mat @ qn) / norms
        order = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in order]
