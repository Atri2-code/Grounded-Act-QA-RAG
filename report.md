# Results

Corpus: MVA, NIA, 412 sections, 1087 chunks. Embedder: lsa (offline LSA, not a neural model). Seed 42.

Questions: 68 (55 answerable, 13 unanswerable); 23 dev (used only to tune abstention thresholds), 45 test.

## Retrieval (all answerable questions, section-level)

| Retriever | hit@1 | hit@3 | hit@3 95% CI | hit@5 | MRR |
|---|---|---|---|---|---|
| bm25 | 52.7% | 72.7% | 61.8% to 83.6% | 78.2% | 0.650 |
| dense | 54.5% | 74.5% | 61.8% to 85.5% | 78.2% | 0.650 |
| hybrid | 56.4% | 70.9% | 58.2% to 81.8% | 78.2% | 0.664 |

Test split only (answerable questions, never used for tuning):

| Retriever | n | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| bm25 | 39 | 46.2% | 69.2% | 76.9% | 0.605 |
| dense | 39 | 46.2% | 74.4% | 79.5% | 0.602 |
| hybrid | 39 | 48.7% | 69.2% | 79.5% | 0.619 |

Paired bootstrap, hybrid_minus_bm25_hit@3: -0.018 (95% interval -0.109 to +0.073).

Paired bootstrap, hybrid_minus_dense_hit@3: -0.036 (95% interval -0.091 to +0.000).

Paired bootstrap, hybrid_minus_bm25_mrr: +0.014 (95% interval -0.040 to +0.075).

## Chunk size (hybrid)

| max chars | chunks | hit@3 | MRR |
|---|---|---|---|
| 300 | 1638 | 74.5% | 0.690 |
| 700 | 1087 | 70.9% | 0.664 |
| 1500 | 609 | 72.7% | 0.673 |

## Answers on the test split (hybrid retrieval, extractive answerer)

Abstention thresholds tuned on dev only: dense similarity >= 0.00, query-term coverage >= 0.70, 3 sentence(s) returned (objective: correct minus 2 x wrong).

| Metric | With abstention | Without abstention |
|---|---|---|
| Answerable questions answered correctly (right section and key fact) | 20.5% | 46.2% |
| Answerable questions answered at all | 28.2% | 100.0% |
| Right section cited, among answered answerable | 81.8% | 48.7% |
| Unanswerable questions correctly refused | 100.0% | 0.0% |
| Wrong answers among all answers given | 27.3% | 60.0% |

Test set: 39 answerable, 6 unanswerable. Unanswerable questions wrongly answered: 0 with abstention, 6 without. Samples are small, so treat differences of a few points as noise.
