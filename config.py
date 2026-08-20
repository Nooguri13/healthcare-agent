"""Central configuration.

Everything is overridable via environment variables so the same code runs
locally with Ollama, in CI with the offline mock, or against a different
open model without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = Path(os.environ.get("HCA_DOCUMENTS_DIR", ROOT / "data" / "documents"))
EVAL_DIR = Path(os.environ.get("HCA_EVAL_DIR", ROOT / "data" / "eval"))
INDEX_DIR = Path(os.environ.get("HCA_INDEX_DIR", ROOT / ".index"))
DB_PATH = Path(os.environ.get("HCA_DB_PATH", ROOT / ".index" / "extractions.db"))

# --- LLM (local / open model via Ollama by default) --------------------------
# Set HCA_LLM_BACKEND=mock to run fully offline (deterministic, no model needed).
LLM_BACKEND = os.environ.get("HCA_LLM_BACKEND", "auto")  # auto | ollama | mock
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("HCA_LLM_MODEL", "llama3.1:8b")

# --- Embeddings --------------------------------------------------------------
# Uses sentence-transformers when installed; falls back to a hashing embedder
# so retrieval still works offline for tests and demos.
EMBED_MODEL = os.environ.get("HCA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Retrieval ---------------------------------------------------------------
CHUNK_SIZE = int(os.environ.get("HCA_CHUNK_SIZE", "700"))      # characters
CHUNK_OVERLAP = int(os.environ.get("HCA_CHUNK_OVERLAP", "120"))
TOP_K = int(os.environ.get("HCA_TOP_K", "4"))
