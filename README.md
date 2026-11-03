# RAG V2 — Document Q&A, evaluation and Words Analytics

A FastAPI application for retrieval-grounded Q&A over two local corpora. Retrieval is local; optional answer generation uses an OpenAI-compatible LLM API (configured through OpenRouter). The UI includes document management, retrieval evaluation, no-context baselines and Words Analytics.

## Corpora

- **Kestrel KB**: five course-provided Markdown documents and the official 10-question set at `data/questions/kestrel.jsonl`.
- **Own corpus**: the Vellora Duo espresso-machine manual at `docs/vellora-duo-espresso-machine.txt`, with the team question set at `data/questions/own.jsonl`.

Both corpora are loaded from disk at request time. Chunk size and overlap are adjustable in the UI/API. Supported uploads are UTF-8 `.txt` and `.md` files.

## Setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Set the following variables in `.env` to enable answer generation and no-context baselines:

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_api_base_url
LLM_MODEL=your_model_name
```

Retrieval evaluation and Words Analytics work without an API key.

## Run

```bash
python -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Evaluation workflow

1. Open **Evaluation**, select a corpus, and keep its question set fixed while comparing configurations.
2. Run retrieval evaluation at a chosen chunk size, overlap and top-k. Repeat with at least one other configuration. Results are timestamped JSON files in `data/results/`.
3. Relevance is gold-quote based: a retrieved chunk is relevant when it contains an exact quote in the question's `answer_quote` or `evidence[].quote` field. For Kestrel, the source filename is also checked. Unanswerable questions are excluded from retrieval HitRate calculations.
4. The summary reports HitRate@1, @3, @5 and @10 overall and by question type. Set top-k to 10 to inspect the top ten chunks; the saved results include all metrics regardless of the selected top-k.
5. Use **Run no-context baseline** to ask the same questions directly to the configured LLM without retrieved context. The output is saved separately. Manually review these outputs against the gold answers and document questions the model already answers correctly.
6. For manual answer grading, enable **Generate answers too** during evaluation. Grade every generated answer as correct, partial, wrong or unsupported, record a rationale, and identify retrieval versus generation failures. The application does not automatically assign subjective grades.

The Kestrel question set uses the course-provided JSONL format with `evidence` arrays. The own question set currently uses `answer_quote`; the evaluator supports both.

## Words Analytics

Open **Words** and select a corpus and chunking configuration.

- **Corpus mode**: load the vocabulary, filter for a word, and select it to inspect its occurrence count, number of chunks and matching chunk text.
- **Question mode**: paste a question to see which tokens occur in the selected corpus, the chunks containing each present word, and the absent tokens. Use this to investigate retrieval misses and record two diagnoses in `REPORT.md`.

## Document management

Open **Documents**, choose a corpus, and upload UTF-8 `.txt` or `.md` files. Documents are stored under `data/corpora/<corpus>/` and included in subsequent retrieval/evaluation requests. Deletion is available through `DELETE /api/documents/{corpus}/{name}`.

## API endpoints

- `GET /api/status`, `GET /api/documents?corpus=...`
- `POST /api/documents/{corpus}`, `DELETE /api/documents/{corpus}/{name}`
- `POST /api/ask`
- `POST /api/evaluate`, `POST /api/no-context`
- `GET /api/questions/{corpus}`, `GET /api/results`
- `GET /api/words/{corpus}`, `GET /api/question-words/{corpus}`

## Limitations

- Answer generation requires a working LLM API key and network access; retrieval-only evaluation does not.
- The evaluator uses exact evidence-quote matching. Paraphrased evidence and questions with multiple independently necessary pieces of evidence may need careful interpretation during manual review.
- The UI supports adding documents and API-level deletion; it does not yet expose per-document selection controls.
- PDF/DOCX ingestion, automatic semantic answer grading and automatic no-context correctness classification are not implemented.
- Citation extraction currently has a known issue: citations may appear inside the generated answer while the outer citations field remains empty.