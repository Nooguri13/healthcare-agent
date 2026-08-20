"""Ingest clinical documents: read, chunk, embed, and persist to the vector store."""
from __future__ import annotations

from pathlib import Path
from typing import List

from . import config, embeddings
from .vectorstore import Chunk, VectorStore


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Character-window chunking with overlap. Simple and predictable."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_index(documents_dir: Path | None = None, index_path: Path | None = None) -> VectorStore:
    documents_dir = Path(documents_dir or config.DOCUMENTS_DIR)
    index_path = Path(index_path or (config.INDEX_DIR / "vectors.json"))

    store = VectorStore(index_path)
    texts_to_embed: List[str] = []

    for path in sorted(documents_dir.glob("*.txt")):
        doc_id = path.stem
        raw = path.read_text()
        for i, piece in enumerate(chunk_text(raw, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            store.chunks.append(
                Chunk(
                    id=f"{doc_id}#{i}",
                    doc_id=doc_id,
                    source=path.name,
                    text=piece,
                    chunk_index=i,
                )
            )
            texts_to_embed.append(piece)

    vectors = embeddings.embed(texts_to_embed)
    store.save(vectors)
    return store


if __name__ == "__main__":  # pragma: no cover
    s = build_index()
    print(f"Indexed {len(s.chunks)} chunks using embeddings backend: {embeddings.backend()}")
