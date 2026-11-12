# RAG V3: dense + BM25 + concept graph

A FastAPI application for retrieval-grounded Q&A over two local corpora (**Kestrel KB** and our own **Vellora Duo** manual).
V3 adds **BM25** (both corpora), a **concept graph** (Kestrel) and **RRF fusion**; every retriever can be switched on or off
from the UI, the API and the evaluation command. Answer generation uses any OpenAI-compatible API (configured in `.env`).
Retrieval and evaluation need no API key.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                 # then fill in LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The first dense query downloads `all-MiniLM-L6-v2` (needs network once).

| `.env` variable | Meaning |
|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | answer generation, no-context baseline, graph extraction |
| `DEFAULT_RETRIEVERS` | retrievers ticked by default, e.g. `dense,bm25` (our chosen default, see REPORT.md) |
| `BM25_STOPWORDS` | `1` (default) or `0`: English stop-word list in BM25 tokenization |

## Switching retrievers on and off

| Where | How |
|---|---|
| **UI** | *Ask* and *Evaluation* tabs: tick `dense`, `BM25`, `concept graph`. Graph is disabled on the own corpus (Kestrel only) and until `app/graph_retriever.py` exists. |
| **API** | `POST /api/ask` and `POST /api/evaluate` accept `"retrievers": ["dense","bm25","graph"]` (any subset), plus `rrf_k` (default 60) and `candidates` (default 10: how many results each retriever gives to the fusion, see DECISIONS D-18). `GET /api/config` lists what is available. |
| **Command line** | `--retrievers dense,bm25,graph` (see below) |

One retriever on: its own ranking. Several on: **RRF** over their lists, `score = Σ weight / (k + rank)`.
Every result shows which retriever(s) found it and at which rank (`dense #3 · bm25 #1`). With the graph on, the UI also shows
the path: entities found in the question, edges followed, and the chunks they came from.

## BM25

- Own implementation in `app/bm25.py` (inverted index, k1 = 1.5, b = 0.75).
- Persisted in `data/index/bm25-<corpus>-<size>-<overlap>-<sw0|sw1>.json` (git-ignored). A fingerprint of the chunk texts and
  the tokenizer settings is stored inside; **uploading, deleting, selecting documents, or changing chunk size/overlap rebuilds it
  automatically** on the next query. Force a rebuild: `POST /api/index/{corpus}/rebuild?chunk_size=800&overlap=10`.
- Tokenization: lowercase, accents folded, `KST-5031` indexed as `kst-5031` + `kst` + `5031`, small English stop-word list.
  Details and measurements: `REPORT.md` and `DECISIONS.md`.
- Dense embeddings are cached the same way (`data/index/dense-*.npz`).

## Concept graph (Kestrel)

> **Status: the concept graph is not implemented in this submission** (see `REPORT.md` section 6). There is no graph build command,
> the third required configuration (`dense+bm25+graph`) was not run, and the graph checkbox stays disabled in the UI.
> What is in place for it: the retriever contract below, the switches in the UI/API/evaluation command (`--retrievers ...,graph`),
> the graph-path panel in the UI, and the Kestrel `two_hop` questions with one evidence quote per hop. Dropping an
> `app/graph_retriever.py` that follows the contract enables everything else without further changes.

Integration contract, so the graph plugs into fusion, UI and evaluation: `app/graph_retriever.py` must expose

```python
search(corpus: str, query: str, k: int, chunk_size: int, overlap: int) -> {
    "hits":  [{"id": "<chunk id from core.build_chunks>", "score": float}, ...],   # best first
    "path":  {"entities": [...], "edges": [{"source","relation","target"}, ...], "chunks": ["<chunk id>", ...]},
    "usage": {"llm_calls": int, "prompt_tokens": int, "completion_tokens": int}   # query-time cost, 0 if none
}
```

Chunk ids depend on chunk size and overlap: build the graph with the same chunking as the evaluation
(Kestrel: 800 / 10, the V2 final value).

## Evaluation

```bash
# the three required configurations (+ BM25 alone, BM25 without stop words), both corpora
python -m app.evaluate matrix

# one configuration
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25,graph --top-k 10

# V2 sanity check: dense on the original ten Kestrel questions only
python -m app.evaluate run --corpus kestrel --retrievers dense --ids q01,q02,q03,q04,q05,q06,q07,q08,q09,q10

# weighted RRF, BM25 without stop words, generated answers (LLM calls)
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25,graph --weights dense=1,bm25=1,graph=2
python -m app.evaluate run --corpus own --retrievers bm25 --no-stopwords
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25 --generate
```

Defaults: V2's final chunking per corpus (Kestrel 800/10, own 500/20), `top_k` 10. Dense runs in **strict mode**: if
sentence-transformers cannot load, the command stops instead of silently measuring TF-IDF (`--allow-tfidf-fallback` overrides it
and the summary then carries a warning).

Output in `data/results/v3/`: one JSON per run (every question, per-hop gold chunks, which retriever found what at which rank,
cost) and `summary.md` / `summary.json`: results table (configurations × corpora × question types), cost per question and index
build, and the automatic list of questions where adding a retriever made things worse.

**Metrics.** A question has one *hop* per evidence quote. *Hit@k*: at least one hop has a gold chunk in the top k (V2's metric).
*Recall@k*: share of hops that have a gold chunk in the top k, so it tells whether **all** hops of a chain were retrieved.
`not_in_corpus` questions have no gold chunk and are excluded from both. Gold chunks = chunks containing the quote, same
rule as V2. Types are reported with the assignment's names (`different_vocabulary` → `vocabulary`, `two_hops` → `two_hop`);
the question files are untouched, the original name is kept as `type_raw` in the results.

The UI *Evaluation* tab runs the same code for one configuration.

## Question sets

`data/questions/kestrel.jsonl` (q01–q10 official; **q11, q12 added in V3**, both `two_hop` with a quote per hop) and
`data/questions/own.jsonl`. Old questions are never edited; new ones are appended. `tests/test_questions.py` enforces this.

## Tests

```bash
python -m unittest discover -s tests -t .     # or: python -m pytest -q
```

Covers the tokenizer, BM25 ranking and index persistence/invalidation, RRF, retriever orchestration, dense caching, and the
metrics (per-hop recall, type mapping, regression detection). They use a fake embedding model, so no download is needed.

## Repository map

| Path | Content |
|---|---|
| `app/core.py` | chunking, corpora, dense retriever + caches, answer generation |
| `app/bm25.py` | BM25 index, tokenizer, persistence |
| `app/fusion.py` | RRF |
| `app/retrieval.py` | retriever switches, fusion, timings, graph contract |
| `app/evaluation.py`, `app/evaluate.py` | metrics, CLI |
| `app/main.py`, `app/static/index.html` | API and UI |
| `data/` | corpora, questions, results (`results/v3/` = V3), derived indexes (`index/`, git-ignored) |
| `REPORT.md`, `DECISIONS.md`, `docs/REPORT_V2.md` | V3 report, decision log, previous report |
| `bm25/`, `concepts/`, `ask_*.py` | course exercises (standalone scripts, not used by the app) |
