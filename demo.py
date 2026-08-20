"""End-to-end demo: build the index, ask questions, extract records, run a query.

Run:  python scripts/demo.py
Works offline (mock backend). Set up Ollama for real answers (see README).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embeddings, ingest, llm, store  # noqa: E402
from src import rag as rag_mod  # noqa: E402
from src.extract import extract_and_store  # noqa: E402


def main():
    print(f"LLM backend: {llm.active_backend()} | Embeddings: {embeddings.backend()}\n")

    print("1) Building vector index over discharge summaries...")
    s = ingest.build_index()
    print(f"   indexed {len(s.chunks)} chunks\n")

    print("2) RAG Q&A with citations:")
    for q in [
        "What was the principal diagnosis for patient PT-1001?",
        "What organism grew in the pneumonia patient's blood cultures?",
    ]:
        res = rag_mod.ask(q)
        srcs = ", ".join(res.retrieved_doc_ids[:2])
        print(f"   Q: {q}\n   A: {res.answer}\n   sources: {srcs}\n")

    print("3) Structured extraction -> SQLite:")
    results = extract_and_store()
    ok = sum(r.ok for r in results)
    print(f"   extracted & stored {ok}/{len(results)} records\n")

    print("4) SQL over extracted records (patients over 50 by diagnosis):")
    conn = store.connect()
    rows = store.query(
        conn,
        "SELECT patient_id, age, principal_diagnosis FROM discharge_records "
        "WHERE age > 50 ORDER BY age DESC",
    )
    for r in rows:
        print(f"   {r['patient_id']} (age {r['age']}): {r['principal_diagnosis']}")
    conn.close()


if __name__ == "__main__":
    main()
