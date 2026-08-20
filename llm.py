"""LLM abstraction.

The agent is provider-agnostic behind this thin interface. The default target is
a *local, open* model served by Ollama. When no model server is reachable (CI,
a laptop without Ollama), it transparently falls back to a deterministic mock so
the pipeline, MCP server, and eval harness all still run end to end.

The mock is intentionally simple and rule-based. It is NOT meant to be accurate;
it exists so the plumbing is testable offline. Real answers require Ollama.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional

from . import config


class LLMError(RuntimeError):
    pass


def _ollama_available() -> bool:
    try:
        req = urllib.request.Request(f"{config.OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_generate(prompt: str, system: Optional[str], temperature: float) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system or "",
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()
    except urllib.error.URLError as e:  # pragma: no cover - network dependent
        raise LLMError(f"Ollama request failed: {e}") from e


# --- Deterministic offline mock ---------------------------------------------
_FIELD_PATTERNS = {
    "patient_id": r"Patient ID:\s*([A-Za-z0-9\-]+)",
    "age": r"Age:\s*(\d+)",
    "sex": r"Sex:\s*([MF])",
    "principal_diagnosis": r"Principal Diagnosis:\s*(.+)",
    "attending_physician": r"Attending Physician:\s*(.+)",
    "admission_date": r"Admission Date:\s*([0-9\-]+)",
    "discharge_date": r"Discharge Date:\s*([0-9\-]+)",
    "allergies": r"Allergies:\s*(.+)",
}


def _mock_generate(prompt: str, system: Optional[str]) -> str:
    """Handle the two prompt shapes this app uses: extraction and RAG answering."""
    # Extraction requests ask for JSON; detect via a marker in the prompt.
    if "Return ONLY JSON" in prompt:
        source = prompt.split("<document>", 1)[-1]
        out = {}
        for field, pat in _FIELD_PATTERNS.items():
            m = re.search(pat, source)
            if m:
                val = m.group(1).strip().rstrip(".")
                out[field] = val
        # medications: collect bullet lines under the medications heading
        meds = re.findall(r"-\s*([A-Za-z].+?)(?:\n|$)", source)
        if meds:
            out["medications"] = [m.strip() for m in meds][:12]
        return json.dumps(out)

    # RAG answering: return the most relevant sentence(s) from the provided context.
    question = ""
    qm = re.search(r"Question:\s*(.+)", prompt)
    if qm:
        question = qm.group(1)
    ctx = prompt.split("<context>", 1)[-1].split("</context>", 1)[0]
    q_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", question)}
    best, best_score = "", -1
    for sent in re.split(r"(?<=[.\n])\s+", ctx):
        score = sum(1 for w in re.findall(r"[a-zA-Z]{4,}", sent) if w.lower() in q_words)
        if score > best_score and sent.strip():
            best, best_score = sent.strip(), score
    if best_score <= 0:
        return "I could not find that information in the provided documents."
    return best


def generate(prompt: str, system: Optional[str] = None, temperature: float = 0.0) -> str:
    """Generate a completion using the configured backend."""
    backend = config.LLM_BACKEND
    if backend == "auto":
        backend = "ollama" if _ollama_available() else "mock"
    if backend == "ollama":
        return _ollama_generate(prompt, system, temperature)
    if backend == "mock":
        return _mock_generate(prompt, system)
    raise LLMError(f"Unknown LLM backend: {backend!r}")


def active_backend() -> str:
    if config.LLM_BACKEND == "auto":
        return "ollama" if _ollama_available() else "mock"
    return config.LLM_BACKEND
