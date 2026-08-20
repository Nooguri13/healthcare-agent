"""LLM-based structured extraction from discharge summaries.

Prompts the local model to return JSON, then validates it against the Pydantic
schema before persisting to SQLite. Extraction failures are surfaced, not
swallowed, so the eval harness can measure them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config, llm, store
from .schema import DischargeRecord

EXTRACT_PROMPT = """You extract structured data from a hospital discharge summary.
Return ONLY JSON with these keys:
  patient_id (string), age (int), sex ("M" or "F"), admission_date, discharge_date,
  attending_physician, principal_diagnosis, allergies, medications (array of strings).
Use null when a field is absent. Do not add commentary.

<document>
{document}
</document>
"""


@dataclass
class ExtractionResult:
    doc_id: str
    ok: bool
    record: Optional[DischargeRecord]
    error: Optional[str]
    raw: str


def _parse_json(text: str) -> dict:
    """Tolerant JSON extraction: grab the first {...} block if the model added prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def extract_document(text: str, doc_id: str) -> ExtractionResult:
    prompt = EXTRACT_PROMPT.format(document=text)
    raw = llm.generate(prompt, temperature=0.0)
    try:
        data = _parse_json(raw)
        record = DischargeRecord(**data)
        return ExtractionResult(doc_id=doc_id, ok=True, record=record, error=None, raw=raw)
    except Exception as e:  # validation or parse failure
        return ExtractionResult(doc_id=doc_id, ok=False, record=None, error=str(e), raw=raw)


def extract_and_store(documents_dir: Path | None = None, db_path: Path | None = None) -> list[ExtractionResult]:
    documents_dir = Path(documents_dir or config.DOCUMENTS_DIR)
    conn = store.connect(db_path)
    results = []
    for path in sorted(documents_dir.glob("*.txt")):
        res = extract_document(path.read_text(), path.stem)
        results.append(res)
        if res.ok and res.record is not None:
            store.upsert(conn, res.doc_id, res.record)
    conn.close()
    return results


if __name__ == "__main__":  # pragma: no cover
    for r in extract_and_store():
        status = "OK " if r.ok else "FAIL"
        detail = r.record.principal_diagnosis if r.ok else r.error
        print(f"[{status}] {r.doc_id}: {detail}")
