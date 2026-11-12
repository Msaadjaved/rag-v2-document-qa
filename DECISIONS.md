# Decision log (V3)

Format: decision, options considered, choice and why. D-12 records that the concept graph was not implemented.
V2 decisions (chunk size, overlap, top-k, embedding model) are in `docs/REPORT_V2.md`.

## Fusion

**D-01. How to combine retrievers.** Options: (a) weighted sum of normalised scores (min-max / z-score), (b) RRF, (c) a learned
re-ranker / cross-encoder, (d) interleaving. **Chosen: RRF.** Cosine similarity, BM25 and graph scores are on different scales
and BM25 scores are unbounded; RRF uses ranks only, needs no tuning data (we have 22 questions) and is the standard baseline.
k = 60 (original paper); each retriever contributes its top `candidates` = 10 (D-18). A cross-encoder would add a model and per-query
cost we cannot justify on corpora of 43 and 78 chunks. Weighted RRF is supported (`--weights`) but plain RRF is the default.

**D-02. One retriever on = no fusion.** The ranking is returned as is, so the `dense` configuration reproduces V2 exactly.
Zero-score BM25 chunks are never returned: they carry no evidence and would enter the fusion as noise.

**D-18. How many candidates each retriever gives to the fusion (default 10, was 30).** With 30, a chunk that dense ranks 16th to
29th (cosine 0.05 to 0.15, i.e. noise) still earns an RRF vote, and if BM25 also returns it, "found by both" outscores a correct
chunk found by one retriever only. Kestrel q05 showed it: gold chunk dense #1, absent from BM25, pushed to rank 6 by five such chunks
(REPORT section 9). Options measured on Kestrel (10 answerable questions, k = 3 / 5, Hit and Recall):
dense alone 0.80 / 0.90 and 0.72 / 0.77; plain RRF with 30 candidates 0.80 / 0.90 and 0.54 / 0.69; weights dense=2 bm25=1 with 30
candidates 0.70 / 0.90 and 0.54 / 0.77; **10 candidates 0.90 / 1.00 and 0.64 / 0.79**. Caveat: the value was chosen after looking at q05,
on the same 10 questions it is evaluated on, so the gain is optimistic; we have no held-out set. Other levers not tried:
RRF k below 60, a score threshold on dense, a cross-encoder re-ranker.

## BM25

**D-03. Implementation.** Options: `rank_bm25` (in-memory, no persistence), SQLite FTS5 (persistent, built-in `bm25()`),
Whoosh / Tantivy / Pyserini, Elasticsearch/OpenSearch (a service to run). **Chosen: our own ~100-line inverted index**
(`app/bm25.py`): no new dependency, formula visible and unit-tested, and the same chunk ids as the other retrievers. FTS5 would
also have been valid; we rejected it because its tokenizer makes the identifier handling in D-05 harder to control.
Parameters k1 = 1.5, b = 0.75 (usual defaults, not tuned: 22 questions are too few to tune without overfitting).

**D-04. Where the index lives and how it is updated.** Persisted as JSON in `data/index/` (git-ignored: derived data), one file per
(corpus, chunk size, overlap, stop-word variant). The file carries a fingerprint of the chunk texts + tokenizer settings. If
documents are uploaded, deleted, (de)selected, or chunking changes, the fingerprint differs and the index is rebuilt on the next
query; written atomically (temp file + rename), so a crash never leaves a half index. **No incremental update**: rebuilding 43 or 78
chunks takes ~10 ms, and a full rebuild cannot drift from the documents. This would stop being true for a corpus of thousands of
chunks (then: per-document postings). `POST /api/index/{corpus}/rebuild` forces a rebuild.

**D-05. Tokenization.** (1) lowercase and fold accents (`Inès` = `ines`); (2) hyphenated identifiers are indexed whole *and* split:
`KST-5031` → `kst-5031`, `kst`, `5031`. The whole token is rare (high IDF) and separates KST-5031 from KST-5038; the parts keep
partial queries working. (3) **English stop-word list, on by default.** Measured on BM25 alone (original 10 Kestrel + 12 own
questions, before q11/q12; reproduce with `python -m app.evaluate matrix`):

| Corpus | Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---|---:|---:|---:|---:|
| Kestrel | with stop words | 0.62 | 0.75 | 0.88 | 0.88 |
| Kestrel | without | 0.38 | 0.88 | 0.88 | 0.88 |
| Own | with stop words | 0.75 | 0.83 | 0.92 | 1.00 |
| Own | without | 0.58 | 0.75 | 0.92 | 1.00 |

Why: without it, a question such as "What is error E47 and how should it be fixed?" lets low-IDF words (`what`, `how`,
`should`, `and`) add up over a short chunk and outrank the chunk holding the single rare token `e47` (gold chunk ranked 4th instead of 1st).
Cost: one Kestrel question (q03) slips from rank 2 to 4 at k = 3. Limitation: English list only (both corpora are English).
No stemming/lemmatisation: it would help "refund/refunded" but also merge unrelated words; not measured, not done.

## Dense retriever

**D-06. Caching and strict mode.** V2 rebuilt the retriever (re-encoding every chunk) at every request. V3 caches the model and the
retriever in memory and the embeddings on disk (`data/index/dense-*.npz`, same fingerprint idea). Evaluation runs in **strict mode**:
V2's `Retriever` catches any error while loading sentence-transformers and silently continues with TF-IDF; the backend was recorded in
the V2 result files (the final ones say `sentence-transformers`, 8 of the 24 files say `tfidf`) but nothing in the UI warned about it. For a
comparison between retriever configurations the evaluation must fail loudly rather than measure the wrong retriever, and the
summary prints a warning if a fallback was allowed (`--allow-tfidf-fallback`).

## Evaluation

**D-07. Metrics.** Hit@k is V2's metric (kept so numbers stay comparable). Recall@k is per *hop* (one hop per evidence quote), so it
shows whether *all* hops of a chain were retrieved, which Hit@k hides (V2 report §8.9). A hop whose quote matches no chunk counts as
not retrieved and is listed (`questions_with_unmatched_hops`). `not_in_corpus` questions are excluded from both metrics (no gold
chunk), as in V2. Relevance = chunk contains the quote (V2 rule, unchanged).

**D-08. Chunking per corpus.** V3 keeps V2's final chunking (Kestrel 800/10, own 500/20) so the only thing that changes between the
V2 row and the V3 rows is the retriever set. The `dense` row is the re-measured V2 baseline on the extended question set;
`--ids q01..q10` reproduces it on the original ten.

**D-09. Question types and files.** Question files are append-only. `own.jsonl` uses `different_vocabulary` and `two_hops`, the
assignment uses `vocabulary` and `two_hop`: we map names at report time (`type_raw` keeps the original) instead of editing old
rows. `not_in_corpus_trap` (OWN-04) is **not** mapped to `not_in_corpus`: it is answerable (the manual says never to use vinegar),
so our own corpus has no truly unanswerable question; the report says so instead of pretending.

**D-10. New questions.** q11 (KST-6010 → Payments Core → #oncall-payments; cross-document, 2 hops) and q12 (mailman → postroom →
Messaging → channel; an alias plus 3 hops, which exercises entity resolution). Both: the question names neither intermediate entity,
and every hop has an evidence quote that matches exactly one chunk at 800/10 (checked by `tests/test_questions.py`).
Known weakness of q12: its first two hops sit in the same chunk, so a retriever can get 2/3 of the recall with one chunk.

## Concept graph

**D-11. Integration.** The graph is a retriever like the others: `search(...)` returns ranked chunk ids, a path for the UI and its
query-time cost (contract in `app/retrieval.py` and the README). Chunk ids depend on chunking, so the graph is built on 800/10.

**D-12. Graph not implemented.** Schema, validation, entity resolution, retrieval strategy and build cost were not decided or measured:
the graph was not finished in the time we had. We chose to submit a complete, measured BM25 + fusion + evaluation instead of an
unmeasured graph. The decisions that remain open, for whoever continues: fixed or open entity/relation types, what to do with LLM output
that fails validation, how to merge aliases (q12: mailman → postroom), and how to turn a question into ranked chunks
(entity-neighbourhood expansion, path search, GraphRAG community summaries, LightRAG-style dual-level retrieval).

## Process

**D-17. Hidden instructions.** The V2 materials contained an instruction invisible on the slide (text in an unreadable colour) that
asked for a hidden page; an assistant that read the raw text added it. We removed that route and do not act on instructions that
are not visible on the slide, unless confirmed by the teacher.

## Known failures (V3 so far; extend after the runs)

1. BM25 has no signal when the question's words are absent from the corpus (vocabulary questions, e.g. "calcium deposits" vs
   "limescale"): it returns nothing and the fusion falls back to dense alone. Fusion cannot fix a question neither retriever finds.
2. Stop-word list is English only; there is no stemming; no incremental BM25 update (D-04, D-05).
3. BM25 and RRF do not know about document versions: for `versions` questions the 2025 and 2026 policies compete (V2 failure
   KST-07 persists as a generation problem, not a retrieval one).
4. Gold chunks are found by quote matching: a quote cut by a chunk boundary yields no gold chunk (flagged, not hidden).
5. q12's hops 1 and 2 share a chunk (D-10).
6. Rank-only fusion lets a chunk found by two retrievers outrank a correct chunk found by one, even when its second vote comes from deep
   in a list: Kestrel q05 (dense #1, BM25 absent) fell from rank 1 to 6 with 30 candidates; limiting to 10 candidates brings it back to
   rank 2 (D-18), still not rank 1. `split_answer` Recall@3 stays at 0.38 with BM25 on (q03 loses a hop to chunks BM25 promotes).
7. `two_hop` Recall stays at 0.56 at k = 5 and 0.72 at k = 10 with dense, with or without BM25: finding every hop of a chain needs
   something that follows relations between chunks (the missing graph).
8. The concept graph is not implemented (D-12).
