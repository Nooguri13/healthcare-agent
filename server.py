"""MCP server exposing the healthcare agent's capabilities as tools.

Any MCP-compatible client (Claude Desktop, an agent framework, etc.) can call:
  - ask_documents(question)     -> grounded answer + citations
  - extract_document(doc_id)     -> structured JSON for one discharge summary
  - query_records(sql)           -> read-only SQL over the extracted records table
  - list_documents()             -> available document ids

Run as a stdio MCP server:   python -m mcp_server.server
Requires the `mcp` package (pip install mcp). The core library also works
without MCP installed — this file only wires it up as a server.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingest  # noqa: E402
from src import rag as rag_mod  # noqa: E402
from src import store as store_mod  # noqa: E402
from src.extract import extract_document  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    print("The 'mcp' package is required: pip install mcp", file=sys.stderr)
    raise

mcp = FastMCP("healthcare-agent")


def _ensure_index() -> None:
    index_path = config.INDEX_DIR / "vectors.json"
    if not index_path.exists():
        ingest.build_index()


@mcp.tool()
def list_documents() -> list[str]:
    """List available discharge-summary document ids."""
    return [p.stem for p in sorted(config.DOCUMENTS_DIR.glob("*.txt"))]


@mcp.tool()
def ask_documents(question: str) -> dict:
    """Answer a question grounded in the clinical documents, with citations.

    Returns the answer plus the source documents it was drawn from. The answer is
    restricted to the retrieved context; it will say when information is absent.
    """
    _ensure_index()
    res = rag_mod.ask(question)
    return {
        "question": res.question,
        "answer": res.answer,
        "citations": [
            {"source": c.source, "chunk_id": c.chunk_id, "score": c.score} for c in res.citations
        ],
        "retrieved_doc_ids": res.retrieved_doc_ids,
    }


@mcp.tool()
def extract_record(doc_id: str) -> dict:
    """Extract structured fields (diagnosis, dates, meds, etc.) from one document."""
    path = config.DOCUMENTS_DIR / f"{doc_id}.txt"
    if not path.exists():
        return {"ok": False, "error": f"unknown doc_id: {doc_id}"}
    res = extract_document(path.read_text(), doc_id)
    if not res.ok or res.record is None:
        return {"ok": False, "error": res.error, "raw": res.raw}
    return {"ok": True, "record": res.record.dict_compat()}


@mcp.tool()
def query_records(sql: str) -> list[dict]:
    """Run a read-only SELECT over the extracted `discharge_records` table.

    Only SELECT statements are permitted. Populate the table first by running the
    extraction pipeline (see README) or by calling extract_record on each doc.
    """
    stripped = sql.strip().lower()
    if not stripped.startswith("select"):
        return [{"error": "Only SELECT queries are allowed."}]
    if ";" in sql.rstrip(";"):
        return [{"error": "Multiple statements are not allowed."}]
    conn = store_mod.connect()
    try:
        return store_mod.query(conn, sql)
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
