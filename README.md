# Healthcare Document Agent — a vertical AI agent for clinical discharge summaries

A domain-specific ("vertical") AI agent that reads hospital **discharge summaries**
and lets you (a) ask grounded questions with citations, (b) pull the documents into
clean structured records, and (c) call all of it from any agent over **MCP**. It
runs on a **local, open model** (Ollama + sentence-transformers) — no API keys, no
data leaving the machine — and ships with an **evaluation harness and a written
failure analysis**, because measuring where an AI system breaks is the real skill.

> ⚕️ **Synthetic data only.** Every document in `data/documents/` is fabricated and
> labeled `SYNTHETIC`. This project contains no real patient information (PHI) and
> is a portfolio/engineering demo, not a clinical tool.

---

## Why this project

Vertical AI agents — models specialized to one industry's documents and workflow —
are one of the strongest current signals in the ML job market, and evaluation is
the differentiator most portfolios skip. This repo is built around all four of the
capabilities that show up in those job descriptions:

1. **RAG Q&A with citations** — retrieval-augmented answers grounded in source docs.
2. **Structured extraction** — messy clinical text → validated JSON → SQLite.
3. **Evaluation & failure analysis** — labeled test sets, metrics, and a writeup
   of *why* it fails ([`evals/FINDINGS.md`](evals/FINDINGS.md)).
4. **MCP server** — the agent's tools exposed to any MCP client.

## Architecture

```
                 ┌─────────────────────────────────────────────┐
   documents ──▶ │  ingest → chunk → embed → vector store       │
  (.txt)         └───────────────┬─────────────────────────────┘
                                 │ retrieve top-k
                                 ▼
     question ─────────▶  RAG (grounded answer + citations)  ◀── local LLM (Ollama)
                                 ▲                                    │
                                 │                                    │
   documents ──▶ extraction (LLM → Pydantic validation) ── ▶ SQLite records
                                 ▲                                    ▲
                                 │                                    │
                       ┌─────────┴──────────┐             ┌──────────┴─────────┐
                       │  Evaluation harness │             │     MCP server      │
                       │  (Hit@k, field acc) │             │ ask / extract / SQL │
                       └────────────────────┘             └────────────────────┘
```

Every external dependency has an **offline fallback** so the whole thing runs with
zero setup for review or CI:

| Component  | Preferred (local, open)        | Offline fallback                    |
|------------|--------------------------------|-------------------------------------|
| LLM        | Ollama (`llama3.1:8b`)         | deterministic rule-based mock       |
| Embeddings | `sentence-transformers` MiniLM | hashing bag-of-words embedder       |
| Vector DB  | built-in NumPy cosine store    | (same — no service required)        |

## Quickstart

```bash
pip install -r requirements.txt      # numpy + pydantic are enough to run offline

# 1. See the whole pipeline end to end (works with no model installed)
python scripts/demo.py

# 2. Run the evaluation report
python -m evals.run_eval

# 3. Run the offline test suite
pytest -q
```

### Real answers with a local open model

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b
export HCA_LLM_BACKEND=auto      # uses Ollama when reachable, else mock
python scripts/demo.py
```

For real semantic retrieval, install `sentence-transformers` (in requirements);
it is picked up automatically. Everything is configured through environment
variables — see [`src/config.py`](src/config.py).

## Using it over MCP

The agent is exposed as an MCP server with four tools: `list_documents`,
`ask_documents`, `extract_record`, and `query_records` (read-only SQL).

```bash
pip install mcp
python -m mcp_server.server        # stdio MCP server
```

Wire it into an MCP client (e.g. Claude Desktop) with
[`examples/claude_desktop_config.json`](examples/claude_desktop_config.json).

## What to look at first

- [`evals/FINDINGS.md`](evals/FINDINGS.md) — the failure analysis. Start here.
- [`src/rag.py`](src/rag.py) — grounded answering with citations.
- [`src/extract.py`](src/extract.py) + [`src/schema.py`](src/schema.py) — LLM
  extraction guarded by Pydantic validation.
- [`mcp_server/server.py`](mcp_server/server.py) — the MCP tool surface.

## Project layout

```
data/documents/     synthetic discharge summaries (.txt)
data/eval/          labeled QA set + extraction gold set
src/                ingest, embeddings, vector store, rag, extract, schema, store, llm, config
mcp_server/         MCP stdio server
evals/              eval harness + FINDINGS.md (failure analysis)
scripts/demo.py     end-to-end walkthrough
tests/              offline smoke tests
```

## Roadmap

See the "What I would do next" section of `evals/FINDINGS.md`: hard-negative
refusal tests, adversarial extraction documents, Hit@1 / MRR ranking metrics, and
an LLM-judge faithfulness score.

## License

MIT — see [LICENSE](LICENSE).
