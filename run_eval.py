"""Evaluation harness.

Measures two things the hiring market actually cares about:
  1. Retrieval quality   -> Hit@k: did the right document appear in the top-k?
  2. Answer groundedness  -> did the answer contain the expected fact?
  3. Extraction accuracy   -> per-field exact/normalized match against gold labels.

It also prints a per-case failure table so you can *diagnose*, not just score.
Run: python -m evals.run_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, embeddings, ingest, llm  # noqa: E402
from src import rag as rag_mod  # noqa: E402
from src.extract import extract_and_store  # noqa: E402
from src.schema import DischargeRecord  # noqa: E402


def _norm(s):
    return str(s).strip().lower().rstrip(".") if s is not None else ""


def eval_retrieval_and_qa():
    data = json.loads((config.EVAL_DIR / "qa_testset.json").read_text())
    cases = data["cases"]
    hits, grounded, rows = 0, 0, []
    for c in cases:
        res = rag_mod.ask(c["question"])
        retrieved = res.retrieved_doc_ids
        hit = c["expected_doc_id"] in retrieved
        ans_ok = all(_norm(sub) in _norm(res.answer) for sub in c["answer_contains"])
        hits += hit
        grounded += ans_ok
        rows.append((c["id"], hit, ans_ok, c["expected_doc_id"], retrieved[0] if retrieved else "-"))
    return cases, hits, grounded, rows


def eval_extraction():
    gold = json.loads((config.EVAL_DIR / "extraction_goldset.json").read_text())["cases"]
    results = extract_and_store()
    by_id = {r.doc_id: r for r in results}
    total_fields, correct_fields, parse_fail, rows = 0, 0, 0, []
    for g in gold:
        r = by_id.get(g["doc_id"])
        if r is None or not r.ok:
            parse_fail += 1
            rows.append((g["doc_id"], "PARSE/VALIDATION FAIL", r.error if r else "missing"))
            total_fields += len(g["expected"])
            continue
        rec = r.record.dict_compat()
        wrong = []
        for field, exp in g["expected"].items():
            total_fields += 1
            got = rec.get(field)
            if _norm(got) == _norm(exp):
                correct_fields += 1
            else:
                wrong.append(f"{field}: got={got!r} exp={exp!r}")
        rows.append((g["doc_id"], "OK" if not wrong else "FIELD MISMATCH", "; ".join(wrong) or "-"))
    return len(gold), correct_fields, total_fields, parse_fail, rows


def main():
    print("=" * 70)
    print("Healthcare Agent — Evaluation Report")
    print(f"LLM backend: {llm.active_backend()}   Embeddings backend: {embeddings.backend()}")
    print("=" * 70)

    print("\n[1/2] Building index...")
    ingest.build_index()

    cases, hits, grounded, qa_rows = eval_retrieval_and_qa()
    n = len(cases)
    print("\n--- Retrieval & Grounded-Answer ---")
    print(f"Hit@{config.TOP_K}:        {hits}/{n}  ({hits / n:.0%})")
    print(f"Answer grounded:  {grounded}/{n}  ({grounded / n:.0%})")
    print(f"{'case':<6}{'hit':<5}{'answ':<6}{'expected':<16}top1")
    for cid, hit, ans_ok, exp, top1 in qa_rows:
        print(f"{cid:<6}{'Y' if hit else 'N':<5}{'Y' if ans_ok else 'N':<6}{exp:<16}{top1}")

    ndoc, correct, total, parse_fail, ex_rows = eval_extraction()
    print("\n--- Structured Extraction ---")
    print(f"Documents:            {ndoc}")
    print(f"Parse/validation fail: {parse_fail}")
    print(f"Field accuracy:        {correct}/{total}  ({(correct / total) if total else 0:.0%})")
    for doc_id, status, detail in ex_rows:
        print(f"  {doc_id}: {status}" + (f"  ({detail})" if detail != "-" else ""))

    print("\nSee evals/FINDINGS.md for the failure analysis writeup.")


if __name__ == "__main__":
    main()
