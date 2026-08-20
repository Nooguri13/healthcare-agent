"""Retrieval-augmented Q&A with inline citations.

The answer is grounded strictly in retrieved chunks, and every answer carries the
source documents it drew from so a reviewer can verify each claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from . import config, embeddings, llm
from .vectorstore import Chunk, VectorStore

SYSTEM = (
    "You are a careful clinical documentation assistant. Answer ONLY using the "
    "provided context from discharge summaries. If the answer is not in the "
    "context, say you cannot find it. Never invent clinical facts. Be concise."
)

PROMPT_TEMPLATE = """Use the context to answer the question. Cite nothing outside it.

<context>
{context}
</context>

Question: {question}

Answer:"""


@dataclass
class Citation:
    doc_id: str
    source: str
    chunk_id: str
    score: float


@dataclass
class RAGResult:
    question: str
    answer: str
    citations: List[Citation]
    retrieved_doc_ids: List[str]


def _load_store(index_path: Path | None) -> VectorStore:
    index_path = Path(index_path or (config.INDEX_DIR / "vectors.json"))
    store = VectorStore(index_path)
    store.load()
    return store


def ask(question: str, top_k: int | None = None, index_path: Path | None = None) -> RAGResult:
    top_k = top_k or config.TOP_K
    store = _load_store(index_path)
    q_emb = embeddings.embed_one(question)
    hits = store.search(q_emb, top_k)

    context_blocks, citations, doc_ids = [], [], []
    for chunk, score in hits:
        context_blocks.append(f"[source: {chunk.source} | {chunk.id}]\n{chunk.text}")
        citations.append(
            Citation(doc_id=chunk.doc_id, source=chunk.source, chunk_id=chunk.id, score=round(score, 4))
        )
        if chunk.doc_id not in doc_ids:
            doc_ids.append(chunk.doc_id)

    prompt = PROMPT_TEMPLATE.format(context="\n\n".join(context_blocks), question=question)
    answer = llm.generate(prompt, system=SYSTEM, temperature=0.0)

    return RAGResult(question=question, answer=answer, citations=citations, retrieved_doc_ids=doc_ids)


if __name__ == "__main__":  # pragma: no cover
    import sys

    q = " ".join(sys.argv[1:]) or "What medication was prescribed for the heart failure patient?"
    res = ask(q)
    print(f"Q: {res.question}\n")
    print(f"A: {res.answer}\n")
    print("Citations:")
    for c in res.citations:
        print(f"  - {c.source} ({c.chunk_id})  score={c.score}")
