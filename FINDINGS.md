# Failure Analysis

This is the part that matters. Building a RAG demo that answers one question is
easy; the engineering signal is in *measuring where it breaks and knowing why*.
Below are findings from `python -m evals.run_eval` on the 8-question QA set and
the 6-document extraction gold set.

## Headline numbers

| Configuration | Retrieval Hit@4 | Grounded answer | Extraction field acc. |
|---|---|---|---|
| Offline fallback (hashing embeddings + mock LLM) | 100% (8/8) | 12% (1/8) | 100% (36/36) |
| Local model (sentence-transformers + Ollama `llama3.1:8b`) | run `make eval` to fill in | — | — |

> The offline column is what CI runs with no model server. The local-model row
> is what you reproduce on a laptop with Ollama; numbers vary by model.

## Finding 1 — Retrieval success does NOT equal answer success

The single most important observation: **Hit@4 was a perfect 100%, yet grounded
answers were 12%.** The correct document was retrieved every time, but the answer
was still usually wrong. This decouples two failure modes that beginners conflate:

- *Retrieval failure* — the evidence never reaches the model.
- *Generation failure* — the evidence is present but the model doesn't use it.

Here, generation was the bottleneck, not retrieval. Reporting a single
"accuracy" number would have hidden that. **Lesson: always measure retrieval and
generation separately.**

## Finding 2 — Ranking quality is invisible to Hit@k

Look at the `top1` column in the report: for qa1, qa2, qa6, qa7 the *correct*
document was within the top 4 (so Hit@4 passed) but a generic document
(`discharge_004`) was ranked #1. The hashing fallback embedder captures
lexical overlap but not meaning, so a query about "blood cultures" or "heart
failure" still surfaces a hypertension note near the top.

Because the mock answerer leans on the highest-ranked context, bad ranking
directly caused bad answers. **Hit@4 looked perfect while Hit@1 was poor** — a
classic metric-choice trap. Switching to real sentence-transformer embeddings is
expected to fix ranking; that is the first thing to verify with `make eval`.

## Finding 3 — Extraction is easy on clean text, and that is a warning

Structured extraction scored 100% field accuracy, but every source here is a
*well-formatted* discharge summary with labeled fields. This number will not hold
on real-world inputs. Known fragilities not yet exercised by the gold set:

- **Free-text diagnoses** buried in prose rather than after a `Principal
  Diagnosis:` label.
- **Date formats** other than `YYYY-MM-DD` (e.g. `03/02/2026`, `Mar 2, 2026`).
- **Missing fields** — the schema allows nulls, but the model may hallucinate a
  plausible value instead of returning null. This is the most dangerous failure
  in a clinical setting and deserves a dedicated negative test.
- **Medication parsing** — dose/route/frequency are currently captured as one
  string; splitting them is future work.

**Lesson: a high score on clean data mostly measures the data, not the system.**
The next iteration should add adversarial documents (unlabeled, reformatted,
partially redacted) to the gold set.

## Finding 4 — Validation caught nothing here, which is the point

`parse/validation fail = 0` is a *good* zero: Pydantic accepted every record. But
the guardrail matters most when the model misbehaves. The `sex` validator already
normalizes and rejects out-of-domain values, and `age` is bounded to 0–120. With
a real LLM at non-zero temperature, expect occasional malformed JSON — the
tolerant parser in `extract._parse_json` and the schema are what keep those out
of the database instead of silently corrupting it.

## What I would do next

1. Run `make eval` against Ollama and record the local-model row above.
2. Add a hard negative to the QA set (a question the documents *cannot* answer)
   and score refusal rate — grounded systems must say "not found."
3. Expand the extraction gold set with messy/adversarial documents.
4. Add Hit@1 and Mean Reciprocal Rank alongside Hit@k to expose ranking quality.
5. Track answer faithfulness with a second-pass LLM judge (or RAGAS) once a
   model server is available.
