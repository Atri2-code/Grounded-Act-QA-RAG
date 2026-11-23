# Grounded Act QA: retrieval-augmented question answering over Indian Acts, with an evaluation harness

A small, honest RAG system for answering questions about statutes. It retrieves the right section, answers only from that text, cites the section, and refuses when the evidence is weak. The point of the project is not the answerer, it is the **evaluation**: a labelled question set, a dev/test split, retrieval ablations with confidence intervals, and a failure analysis, so you can see what works and what does not.

> A fluent wrong answer is worse than no answer. The system is built around that rule: answers must be quotable from the Act, and out-of-scope questions get "I could not find this".

This is a portfolio project, not legal advice. The Act text is a snapshot and may not include the latest amendments.

## What is in it

| Piece | Where | What it does |
|---|---|---|
| Ingestion and chunking | `src/actqa/data.py`, `chunking.py` | Loads Act JSON, strips amendment footnote markers, splits each section into sentence-aligned chunks that never cross a section boundary |
| Keyword retrieval | `lexical.py` | BM25 with stop words and Porter stemming |
| Vector database | `embeddings.py`, `vectorstore.py` | Dense vectors stored and searched in **ChromaDB** (cosine). Default embedder is offline LSA (TF-IDF + truncated SVD); a neural embedder (`sentence-transformers`) is a one-line swap |
| Hybrid retrieval | `retrieval.py` | BM25 and dense rankings fused with reciprocal rank fusion, then deduplicated to sections |
| Grounded answering | `answer.py` | Extractive answerer: returns sentences from the cited section, abstains when query-term coverage is low |
| Generative answerer | `llm.py` | Optional Anthropic API answerer. Must cite, must say `NOT_FOUND`, and every answer is checked: citations must be among the retrieved sections and each sentence must be grounded in them, otherwise it becomes an abstention |
| LangChain adapter | `langchain_adapter.py` | `ActRetriever` exposes the pipeline as a LangChain retriever |
| API | `api.py` | FastAPI: `POST /ask`, `GET /health`, validated input |
| Evaluation | `evaluate.py`, `eval/questions.jsonl` | 68 labelled questions, dev/test split, ablations, bootstrap intervals, failure report |
| Tests | `tests/` | 35 offline unit tests on a synthetic mini-statute |

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/fetch_corpus.py              # downloads the two Acts into ./data
python -m actqa ask "What is the fine for not wearing a seat belt?"
python -m actqa eval                         # regenerates results/metrics.json, report.md, failures.md
python -m pytest                             # 35 tests, no network needed
uvicorn actqa.api:app                        # then POST {"question": "..."} to /ask
```

The corpus is **not** committed. `scripts/fetch_corpus.py` downloads the Negotiable Instruments Act, 1881 (156 sections) and the Motor Vehicles Act, 1988 (256 sections) from [civictech-India/Indian-Law-Penal-Code-Json](https://github.com/civictech-India/Indian-Law-Penal-Code-Json), a public dataset of bare Acts originally published on India Code.

## How the evaluation works

- **68 questions**: 55 answerable (each with the gold section(s) and a key fact that must appear in the answer) and 13 unanswerable (10 clearly out of scope, 3 near-misses such as "what is the licence fee amount", which the Act leaves to rules).
- **Gold labels are checked by a test**: every expected fact must appear verbatim in its gold section (`tests/test_evaluate.py`).
- **Dev/test split**: 23 dev, 45 test. The dev split is used only to choose the abstention thresholds and the number of sentences returned. Nothing else was tuned on the questions. Retrieval design choices (adding stemming) were made after looking at dev failures only.
- **Metrics**: section-level hit@k and MRR for retrieval; for answers, whether the right section was cited and the key fact is in the answer, how many unanswerable questions were refused, and what share of the answers given were wrong. The tuning objective is `correct - 2 x wrong`, because a wrong answer costs more than a refusal.
- **Statistics**: percentile bootstrap intervals over questions, and paired bootstrap for comparing retrievers.

## Results (seed 42; regenerate with `python -m actqa eval`)

Corpus: 2 Acts, 412 sections, 1,087 chunks. Full tables are in `results/report.md`.

**Retrieval, all 55 answerable questions (section-level):**

| Retriever | hit@1 | hit@3 | hit@3 95% CI | hit@5 | MRR |
|---|---|---|---|---|---|
| BM25 | 52.7% | 72.7% | 61.8% to 83.6% | 78.2% | 0.650 |
| Dense (LSA in ChromaDB) | 54.5% | 74.5% | 61.8% to 85.5% | 78.2% | 0.650 |
| Hybrid (RRF) | 56.4% | 70.9% | 58.2% to 81.8% | 78.2% | 0.664 |

On the 39 answerable **test** questions only: BM25 69.2%, dense 74.4%, hybrid 69.2% hit@3.

**Answers on the test split (45 questions: 39 answerable, 6 unanswerable):**

| Metric | With abstention | Without abstention |
|---|---|---|
| Answerable questions answered correctly | 20.5% | 46.2% |
| Answerable questions answered at all | 28.2% | 100% |
| Right section cited, among answered answerable | 81.8% | 48.7% |
| Unanswerable questions refused | 6 of 6 | 0 of 6 |
| Wrong answers among all answers given | 27.3% (3 of 11) | 60.0% |

### What the numbers say

- **Hybrid retrieval did not beat either single retriever.** The paired bootstrap intervals include zero, so on 55 questions the three are statistically indistinguishable. Fusing rankings is not automatically better.
- **Abstention trades coverage for trust.** It cut the share of wrong answers from 60% to 27% and refused every out-of-scope test question, but it answers only 28% of answerable questions. Whether that trade is right depends on the cost of a wrong answer, which is why the objective weights wrong answers twice.
- **The failure pattern is vocabulary, not ranking.** Penalty questions ("what is the penalty for driving without insurance?") fail because the section says "whoever drives a motor vehicle in contravention of section 146 shall be punishable", so it never mentions insurance. A neural embedder, query rewriting or a reranker is the obvious fix. See `results/failures.md`.
- **One "wrong" answer is arguably defensible.** "Is a vehicle allowed to run without registration?" cited s.192 (the penalty for using an unregistered vehicle) while the gold label is s.39 (the requirement). Gold labels are strict and were written by one person.
- **Chunk size barely matters** (300 / 700 / 1500 characters: 74.5% / 70.9% / 72.7% hit@3), well inside the noise, so the default was not changed.

## Limitations

- The dense embedder is **offline LSA**, chosen because no model download was possible where this was built. It is not a neural embedding model, so the dense numbers say little about what a modern embedder would do. Swap in `sentence-transformers` with `Pipeline(..., embedder="st:sentence-transformers/all-MiniLM-L6-v2")` and re-run the evaluation.
- The **generative answerer** (`llm.py`) is implemented and unit-tested with a fake client, but it was **not run against the real API** for these results, so there are no measured numbers for it.
- 68 questions from one author is small. Differences of a few points are noise.
- Act text is a dataset snapshot and may lag amendments. Not legal advice.
- The API has no authentication or rate limiting.
- Re-running the evaluation reproduces every number except a difference of about 0.0002 in one chunk-size ablation cell, probably from tie-breaking in ChromaDB's approximate vector search (not confirmed).

## Next steps

1. Run the evaluation with a neural embedder and a cross-encoder reranker and compare.
2. Run `LLMAnswerer` on the test split and report the grounded-citation rate and refusal behaviour.
3. Add query rewriting for penalty-style questions.
4. Grow the question set and add a second labeller to measure label agreement.

## Layout

```
src/actqa/        pipeline, retrieval, answering, evaluation, API, LangChain adapter
eval/questions.jsonl   68 labelled questions with dev/test split
results/          metrics.json, report.md, failures.md (generated)
scripts/fetch_corpus.py
tests/            35 offline tests
```
