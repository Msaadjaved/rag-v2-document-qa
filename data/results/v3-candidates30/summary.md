# V3 evaluation summary

_Generated 2026-11-11 04:40:35. Chunking per corpus: kestrel 800/10, own 500/20. Hit@k = at least one hop retrieved; Recall@k = share of hops retrieved. not_in_corpus questions are excluded from both._

## Results: configurations × corpora × question types

| Corpus | Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kestrel | dense | **ALL** | 10 | 0.50 | 0.80 | 0.90 | 1.00 | 0.33 | 0.72 | 0.77 | 0.92 |
| kestrel | dense | exact_identifier | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.25 | 0.50 | 0.50 | 1.00 |
| kestrel | dense | split_answer | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 | 1.00 | 1.00 |
| kestrel | dense | vocabulary | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense | versions | 2 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense | two_hop | 3 | 0.33 | 0.67 | 1.00 | 1.00 | 0.17 | 0.39 | 0.56 | 0.72 |
| kestrel | dense+bm25 | **ALL** | 10 | 0.50 | 0.80 | 0.90 | 1.00 | 0.32 | 0.54 | 0.69 | 0.92 |
| kestrel | dense+bm25 | exact_identifier | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.25 | 0.50 | 1.00 | 1.00 |
| kestrel | dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.62 | 1.00 |
| kestrel | dense+bm25 | vocabulary | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| kestrel | dense+bm25 | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25 | two_hop | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.39 | 0.56 | 0.56 | 0.72 |
| own | dense | **ALL** | 12 | 0.67 | 0.92 | 1.00 | 1.00 | 0.67 | 0.92 | 1.00 | 1.00 |
| own | dense | exact_identifier | 5 | 0.60 | 0.80 | 1.00 | 1.00 | 0.60 | 0.80 | 1.00 | 1.00 |
| own | dense | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| own | dense | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| own | dense | two_hop | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| own | dense | not_in_corpus_trap | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | **ALL** | 12 | 0.75 | 0.92 | 1.00 | 1.00 | 0.75 | 0.92 | 1.00 | 1.00 |
| own | dense+bm25 | exact_identifier | 5 | 0.80 | 0.80 | 1.00 | 1.00 | 0.80 | 0.80 | 1.00 | 1.00 |
| own | dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | two_hop | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | not_in_corpus_trap | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Cost per question and index build

| Corpus | Configuration | Retrieval ms/question | Total ms/question | LLM calls/question | Prompt tok/question | Completion tok/question | Dense backend | Index build |
|---|---|---:|---:|---:|---:|---:|---|---|
| kestrel | dense | 31.2 | 31.2 | 0.00 | 0 | 0 | sentence-transformers | dense: 42.12s |
| kestrel | dense+bm25 | 40.5 | 40.5 | 0.00 | 0 | 0 | sentence-transformers | dense: 42.12s; bm25: 0.03s |
| own | dense | 29.2 | 29.2 | 0.00 | 0 | 0 | sentence-transformers | dense: 3.08s |
| own | dense+bm25 | 30.4 | 30.4 | 0.00 | 0 | 0 | sentence-transformers | dense: 3.08s; bm25: 0.04s |

## Where adding a retriever made things worse

**kestrel: `dense` → `dense+bm25`**: 2 question(s) where the first gold chunk ranks worse.
- `q03` (split_answer): rank 1 → 2 (recall@k 1.00 → 1.00) — What happens, step by step, when a customer's card payment fails?
- `q05` (vocabulary): rank 1 → 6 (recall@k 1.00 → 1.00) — I lost my phone and I can't log in anymore. What can I do?

**own: `dense` → `dense+bm25`**: 2 question(s) where the first gold chunk ranks worse.
- `OWN-01` (exact_identifier): rank 4 → 5 (recall@k 1.00 → 1.00) — What is the capacity of the Vellora Duo water tank?
- `OWN-02` (split_answer): rank 2 → 3 (recall@k 1.00 → 1.00) — How long should the water filter be soaked before installation, and how long does it last?
