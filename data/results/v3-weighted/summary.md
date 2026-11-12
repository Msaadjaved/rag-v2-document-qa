# V3 evaluation summary

_Generated 2026-11-11 04:45:48. Chunking per corpus: kestrel 800/10, own 500/20. Hit@k = at least one hop retrieved; Recall@k = share of hops retrieved. not_in_corpus questions are excluded from both._

## Results: configurations × corpora × question types

| Corpus | Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kestrel | dense+bm25 (weighted dense=2,bm25=1) | **ALL** | 10 | 0.60 | 0.70 | 0.90 | 1.00 | 0.34 | 0.54 | 0.77 | 0.92 |
| kestrel | dense+bm25 (weighted dense=2,bm25=1) | exact_identifier | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.25 | 0.50 | 1.00 | 1.00 |
| kestrel | dense+bm25 (weighted dense=2,bm25=1) | split_answer | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.38 | 0.62 | 1.00 | 1.00 |
| kestrel | dense+bm25 (weighted dense=2,bm25=1) | vocabulary | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| kestrel | dense+bm25 (weighted dense=2,bm25=1) | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25 (weighted dense=2,bm25=1) | two_hop | 3 | 0.67 | 0.67 | 1.00 | 1.00 | 0.39 | 0.39 | 0.56 | 0.72 |

## Cost per question and index build

| Corpus | Configuration | Retrieval ms/question | Total ms/question | LLM calls/question | Prompt tok/question | Completion tok/question | Dense backend | Index build |
|---|---|---:|---:|---:|---:|---:|---|---|
| kestrel | dense+bm25 (weighted dense=2,bm25=1) | 28.3 | 28.3 | 0.00 | 0 | 0 | sentence-transformers | dense: 33.04s (cached); bm25: 0.00s (cached) |

## Where adding a retriever made things worse

(no comparable pairs were run)
