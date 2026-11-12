# RAG V3: Team Evaluation Report

## 1. Team and release

- Team members:
  - **Sultan AMIOUR**
  - **Muhammad Saad JAVED**
- Team of 2 (GitHub: `s-amiour`, `Msaadjaved`); the "teams of 3" extra piece does not apply.
- Repository: https://github.com/Msaadjaved/rag-v2-document-qa · tag `v3`
- Previous report: `docs/REPORT_V2.md` · decision log: `DECISIONS.md` · how to run: `README.md`

## 2. Summary

- V3 adds **BM25** (both corpora, persistent index), **RRF fusion** with every retriever switchable from the UI, the API and the
  evaluation command, and **Recall@k per hop** next to V2's HitRate@k.
- **The concept graph is not part of this submission** (section 6). The third required configuration (`dense+bm25+graph`) was not run.
- **BM25 helps, but only once fusion is configured carefully.** On Kestrel, plain RRF with 30 candidates per retriever left Hit@k unchanged
  (0.50 / 0.80 / 0.90 / 1.00 at k = 1, 3, 5, 10) and lowered Recall@3 from 0.72 to 0.54. Limiting each retriever to its 10 best candidates
  (D-18) gives Hit@3 **0.90**, Hit@5 **1.00** and Recall@5 **0.79**, the best of all configurations at k = 3 and 5. It fixes
  `exact_identifier` (Hit@5 0.50 → 1.00). It costs a little on rank 1 (Hit@1 stays 0.50, q05 and q03 drop from rank 1 to 2) and on
  `split_answer` Recall@3 (1.00 → 0.38).
- On our own corpus: Hit@1 0.67 → 0.75, no loss at k ≥ 3, identical with 10 or 30 candidates.
- BM25 needs no LLM call, builds in 0.03 to 0.04 s and adds no measurable query latency (15 to 20 ms per query in every configuration).
- **Limit of this work:** `two_hop` Recall stays at 0.56 at k = 5 and 0.72 at k = 10 with or without BM25. Finding every hop of a chain is
  the unsolved part, and it is what the graph was meant to address.
- **Caveat:** Kestrel has 10 answerable questions, so one question is worth 0.10; most types have n = 1 to 5. Differences are indicative.
  The `candidates` = 10 setting was chosen after looking at q05 and is scored on the same questions (no held-out set).

## 3. Results

Chunking per corpus: Kestrel 800/10, own 500/20 (V2 finals). Dense backend in every run: `sentence-transformers` (all-MiniLM-L6-v2),
embeddings cached on disk. Hit@k = at least one hop retrieved; Recall@k = share of hops retrieved. `not_in_corpus` questions are excluded
from both metrics. Kestrel now has 12 questions (q11 and q12 added; q09 and q10 are `not_in_corpus`), so n = 10 answerable (8 in V2).

**V2 check:** `--ids q01..q10` with `dense` reproduces V2 exactly (Hit@1/3/5/10 = 0.62 / 0.88 / 0.88 / 1.00, same per type), so the `dense`
row is V2's baseline re-measured on the extended question set and nothing else changed.

`candidates` = how many results each retriever gives to the fusion. 30 in the first runs (`data/results/v3-candidates30/`,
`data/results/v3-weighted/`), **10 by default since D-18** (`data/results/v3/`, which this table reports).

### Kestrel (800/10)

| Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense (V2 baseline) | **ALL** | 10 | 0.50 | 0.80 | 0.90 | 1.00 | 0.33 | 0.72 | 0.77 | 0.92 |
| dense | exact_identifier | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.25 | 0.50 | 0.50 | 1.00 |
| dense | split_answer | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 | 1.00 | 1.00 |
| dense | vocabulary | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense | versions | 2 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| dense | two_hop | 3 | 0.33 | 0.67 | 1.00 | 1.00 | 0.17 | 0.39 | 0.56 | 0.72 |
| **dense+bm25 (default)** | **ALL** | 10 | 0.50 | **0.90** | **1.00** | 1.00 | 0.32 | 0.64 | **0.79** | 0.92 |
| dense+bm25 | exact_identifier | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.25 | 0.50 | 1.00 | 1.00 |
| dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.62 | 1.00 |
| dense+bm25 | vocabulary | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | two_hop | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.39 | 0.56 | 0.56 | 0.72 |
| *ablation:* dense+bm25, 30 candidates | **ALL** | 10 | 0.50 | 0.80 | 0.90 | 1.00 | 0.32 | 0.54 | 0.69 | 0.92 |
| *ablation:* dense+bm25, 30 candidates | vocabulary | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| *ablation:* dense+bm25, 30 candidates, weights dense=2 bm25=1 | **ALL** | 10 | 0.60 | 0.70 | 0.90 | 1.00 | 0.34 | 0.54 | 0.77 | 0.92 |
| *ablation:* same, weighted | vocabulary | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| *ablation:* same, weighted | two_hop | 3 | 0.67 | 0.67 | 1.00 | 1.00 | 0.39 | 0.39 | 0.56 | 0.72 |
| dense+bm25+graph | | | not run (section 6) | | | | | | | |

### Own corpus (500/20), identical with 10 and 30 candidates

| Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | **ALL** | 12 | 0.67 | 0.92 | 1.00 | 1.00 | 0.67 | 0.92 | 1.00 | 1.00 |
| dense | exact_identifier | 5 | 0.60 | 0.80 | 1.00 | 1.00 | 0.60 | 0.80 | 1.00 | 1.00 |
| dense | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| dense | two_hop | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense | not_in_corpus_trap | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **dense+bm25 (default)** | **ALL** | 12 | **0.75** | 0.92 | 1.00 | 1.00 | **0.75** | 0.92 | 1.00 | 1.00 |
| dense+bm25 | exact_identifier | 5 | **0.80** | 0.80 | 1.00 | 1.00 | **0.80** | 0.80 | 1.00 | 1.00 |
| dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | two_hop | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | not_in_corpus_trap | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Reading the tables:

- **`exact_identifier`:** BM25 helps on both corpora (Kestrel Hit@5 0.50 → 1.00; own Hit@1 0.60 → 0.80).
- **`vocabulary`:** BM25 cannot help (no shared word) and can hurt through fusion (q05, section 9).
- **`versions`:** Hit@1 0.00 → 0.50 with BM25; Hit@3 is already 1.00 with dense.
- **`two_hop`:** Hit@10 is 1.00 but Recall@10 only 0.72: every chain is "found" but not every hop. In the V2 check, q08 alone has
  Recall@10 = 0.50 with dense: one of its two hops is not in the top 10.
- **`not_in_corpus`:** the two Kestrel questions are excluded from the metrics. The own corpus has none: its `not_in_corpus_trap`
  question (OWN-04, vinegar) is answerable and therefore scored (D-09). OWN-05 (`two_hops`) has a single evidence quote, so Hit = Recall there.
  The own corpus has no `versions` question. Type names in `own.jsonl` (`different_vocabulary`, `two_hops`) are mapped to the assignment's
  names when reporting; the file is untouched.

### Cost per question (no answer generation)

| Corpus | Configuration | Retrieval ms | LLM calls | Index build |
|---|---|---:|---:|---|
| Kestrel | dense | 16.4 | 0 | dense 22.6 s (model load; embeddings cached on disk after the first run) |
| Kestrel | dense+bm25 | 15.2 | 0 | + BM25 0.03 s when built, 0.00 s when loaded from disk |
| Own | dense | 20.0 | 0 | dense 0.02 s (cached) |
| Own | dense+bm25 | 13.0 | 0 | + BM25 0.04 s when built, 0.00 s when loaded |

Latency differences between configurations are within run-to-run noise (an earlier run measured 31 ms and 40 ms for Kestrel
dense and dense+bm25; this one 16 ms and 15 ms). The honest statement is: BM25 adds no LLM call and no measurable per-query cost.
The measured costs of generation (1 LLM call per question) are the same in every configuration.

## 4. Default configuration and why

- **Kestrel: `dense+bm25`, plain RRF, `candidates` = 10, top_k = 5.** Against dense alone: Hit@3 0.80 → 0.90, Hit@5 0.90 → 1.00,
  Recall@5 0.77 → 0.79, and `exact_identifier` Hit@5 0.50 → 1.00, which matters because the Kestrel runbook is organised by KST-xxxx codes.
  Against the other fusion settings it is best at k = 3 and 5 (30 candidates: Hit@5 0.90, Recall@5 0.69; weighted 2:1 with 30 candidates: Hit@3 0.70).
  Price, stated openly: Recall@3 is 0.64 against 0.72 for dense alone (`split_answer` Recall@3 0.38 against 1.00: q03 loses a hop to chunks
  that BM25 promotes), and q05 and q03 sit at rank 2 instead of rank 1 (Hit@1 stays 0.50).
- **Own corpus: `dense+bm25`, same settings.** Hit@1 0.67 → 0.75 (`exact_identifier` 0.60 → 0.80), no loss at any k ≥ 3, no cost.
- Set `DEFAULT_RETRIEVERS=dense,bm25` in `.env`; `candidates` defaults to 10 in code. Weights are an option (`--weights`), not the default,
  because they did not beat plain RRF with 10 candidates and need tuning data we do not have.
- **Caveat:** `candidates` = 10 was chosen after seeing q05, on the questions it is scored on.

## 5. BM25

- **Index:** own inverted index (`app/bm25.py`), k1 = 1.5, b = 0.75 (usual defaults, not tuned), persisted as JSON in `data/index/`
  (git-ignored), one file per corpus, chunking and tokenizer variant, written atomically (D-03, D-04).
- **Tokenization:** lowercase, accent folding, hyphenated identifiers indexed whole and split (`KST-5031` → `kst-5031`, `kst`, `5031`),
  English stop-word list. Stop-word ablation on BM25 alone (Hit@1): Kestrel 0.62 vs 0.38, own 0.75 vs 0.58; one Kestrel question loses at
  k = 3 (D-05). No stemming (not measured).
- **Updates:** a fingerprint of the chunk texts and tokenizer settings is stored in the index file. Upload, delete, selection or chunking
  changes rebuild the index on the next query; there is no incremental update (a rebuild takes 0.03 to 0.04 s at this size).
  `POST /api/index/{corpus}/rebuild` forces it. Tested in `tests/test_bm25.py` (build, reuse, reload from disk, update, removal, corrupt file).
- **Cost:** no LLM call; build 0.03 to 0.04 s; no measurable per-query cost.
- **Where it helps:** identifiers (section 3). **Where it fails:** vocabulary mismatch. q05 ("I lost my phone and I can't log in") shares no
  content word with the guide ("authenticator device", "sign in"): BM25 does not return the gold chunk at all.

## 6. Concept graph

**The concept graph is not implemented in this submission.** It was not finished in the time we had. Consequences, stated plainly:

- no graph build, no schema, validation, entity resolution, graph retrieval or extraction-quality grades;
- the required third configuration (`dense+bm25+graph`) was not run; there is no graph cost, merge or triple analysis;
- the "graph's chunks after fusion" question cannot be answered.

What exists and works toward it: the retriever contract (`app/retrieval.py`, README), the switches in the UI, the API and the evaluation
command (`--retrievers dense,bm25,graph` is accepted and fails with a clear message while `app/graph_retriever.py` is absent), the graph-path
panel in the UI, per-hop Recall@k, and two new Kestrel `two_hop` questions with one evidence quote per hop (q11, 2 hops; q12, 3 hops with an
alias, mailman → postroom → Messaging). The measured gap the graph would have to close is in section 3: `two_hop` Recall@5 = 0.56 and
Recall@10 = 0.72 for dense and for dense+bm25. Open design decisions are listed in D-12.

## 7. The graph in numbers

Not applicable (section 6).

## 8. Fusion: what happened to chunks after RRF

There is no graph, but the same mechanism was measured with BM25. RRF uses ranks only, so a chunk found by two retrievers beats a chunk
found by one, even when the second vote comes from deep in a list. Levers: `candidates`, `--rrf-k`, `--weights`. Measured on Kestrel:

| Setting | Hit@3 | Hit@5 | Recall@3 | Recall@5 | q05 |
|---|---:|---:|---:|---:|---|
| dense alone | 0.80 | 0.90 | 0.72 | 0.77 | rank 1 |
| RRF, 30 candidates | 0.80 | 0.90 | 0.54 | 0.69 | rank 6 |
| RRF, 30 candidates, weights dense=2 bm25=1 | 0.70 | 0.90 | 0.54 | 0.77 | not fixed |
| **RRF, 10 candidates (default)** | **0.90** | **1.00** | 0.64 | **0.79** | **rank 2** |

Not tried: `--rrf-k` below 60, a score threshold on dense, a cross-encoder re-ranker.

## 9. A question where adding a retriever made things worse

**Kestrel q05**, "I lost my phone and I can't log in anymore. What can I do?" (`vocabulary`). Gold chunk `product-guide_md:6`.
Dense ranks it **1st**; BM25 does not return it at all. With plain RRF (k = 60, 30 candidates) it falls to rank **6**: five chunks that
are weak in dense (ranks 2, 16, 22, 27, 29) but present in BM25's top 6 get two votes and outscore it.

| Chunk | Dense rank | BM25 rank | RRF score |
|---|---:|---:|---:|
| support-runbook-errors_md:2 | 2 | 1 | 0.0325 |
| service-catalog_md:3 | 16 | 6 | 0.0283 |
| support-runbook-errors_md:6 | 22 | 3 | 0.0281 |
| billing-policy-2025_md:2 | 27 | 2 | 0.0276 |
| product-guide_md:2 | 29 | 4 | 0.0269 |
| **product-guide_md:6 (gold)** | **1** | none | **0.0164** |

Why: RRF discards scores. Dense ranks 16 to 29 have cosine scores of 0.05 to 0.15 (noise); the gold chunk has 0.41; rank-only fusion
still treats "found by both" as strong evidence, and BM25 matched only generic words here. Weights dense=2 bm25=1 did not fix it.

**Fix, predicted then measured.** With 10 candidates, only chunks inside dense's top 10 keep their dense vote, so four of the five
displacers fall to a single BM25 vote (1/62 to 1/66, below the gold chunk's 1/61). Measured: q05 goes **rank 6 → rank 2** (Hit@3
0.00 → 1.00); Kestrel overall Hit@3 0.80 → 0.90, Hit@5 0.90 → 1.00. It does not change `split_answer` (Recall@3 0.38) or `two_hop`
(Recall@5 0.56), and q05 is still not rank 1. The setting was chosen after looking at this question (D-18).

Other questions that get worse with BM25, all staying in the top 5: Kestrel q03 (`split_answer`) rank 1 → 2; own OWN-01 rank 4 → 5;
own OWN-02 rank 2 → 3. The full automatic list is in `data/results/v3/summary.md`.

## 10. Decision log and known failures

`DECISIONS.md`, D-01 to D-18, and its "Known failures" section. In short: BM25 returns nothing for vocabulary-mismatch questions;
rank-only fusion lets double votes displace a correct single-retriever hit (q05); `split_answer` Recall@3 drops with BM25; `two_hop`
Recall@5 and @10 are unchanged by BM25 (0.56, 0.72); the English-only stop-word list; no incremental BM25 update; gold chunks are found by quote matching
(a quote cut by a chunk boundary yields no gold chunk; none occurs in our question files, checked by `tests/test_questions.py`);
the graph is missing.

## 11. How to reproduce

```bash
pip install -r requirements.txt
cp .env.example .env                                  # LLM_* only needed to generate answers
python -m unittest discover -s tests -t .             # 54 tests, no model download
python -m app.evaluate matrix --required-only         # data/results/v3/ (summary.md = every table above)
python -m app.evaluate run --corpus kestrel --retrievers dense --ids q01,q02,q03,q04,q05,q06,q07,q08,q09,q10   # V2 check
# ablations with 30 candidates:
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25 --candidates 30 --out-dir data/results/v3-candidates30
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25 --candidates 30 --weights dense=2,bm25=1 --out-dir data/results/v3-weighted
```

Dense runs in strict mode: the evaluation stops rather than silently measuring TF-IDF.
