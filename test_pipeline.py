"""Smoke tests that run fully offline (mock LLM + hashing embeddings)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HCA_LLM_BACKEND", "mock")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ingest, rag  # noqa: E402
from src.extract import extract_document  # noqa: E402
from src.schema import DischargeRecord  # noqa: E402


def test_index_builds_and_retrieves():
    ingest.build_index()
    res = rag.ask("What was the principal diagnosis for patient PT-1001?")
    # correct document should be retrieved
    assert "discharge_001" in res.retrieved_doc_ids
    assert res.citations, "expected at least one citation"


def test_extraction_validates():
    doc = (Path(__file__).resolve().parent.parent / "data/documents/discharge_002.txt").read_text()
    res = extract_document(doc, "discharge_002")
    assert res.ok
    assert isinstance(res.record, DischargeRecord)
    assert res.record.patient_id == "PT-1002"
    assert res.record.sex == "F"


def test_schema_rejects_bad_sex():
    rec = DischargeRecord(patient_id="PT-9", sex="unknown")
    assert rec.sex is None  # normalized/rejected, not stored as garbage


def test_schema_bounds_age():
    import pytest

    with pytest.raises(Exception):
        DischargeRecord(patient_id="PT-9", age=999)
