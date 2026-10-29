# RAG V3 — BM25 + Concept graphs

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

## Applications

1. `ask_norag_google.py`
- NoRAG with Google Gemini via OpenAI SDK (`gemini-3.5-flash-lite`).
- Loads `docs/CrimeAndPunishment.txt`, injects a "needle" passage, and runs two direct large-context questions:
	- baseline question about Marmeladov (present in the original book)
	- needle question about Denis (the injected passage)

2. `ask_rag_google.py`
- RAG with Google Gemini for generation via OpenAI SDK.
- Uses local embeddings (`all-MiniLM-L6-v2`) + local ChromaDB (`chroma_db/`) to retrieve relevant chunks before asking the model.
- Currently points to the espresso-machine manual and retrieves `n_results=5` for the question: `What does error E47 mean and how do I fix it?`

3. `ask_bm25_rag_google.py`
- RAG with Google Gemini for generation via OpenAI SDK and in-memory BM25 lexical retrieval.
- See [README_BM25.md](./README_BM25.md) for how it works, how to run it, and the BM25 challenge.

### Run

With your virtual environment activated, run your chosen application:

```bash
# NoRAG + Google
python ask_norag_google.py

# RAG + Google
python ask_rag_google.py

# BM25 RAG + Google
python ask_bm25_rag_google.py
```

## Notes

- There exists two Python files, `ask_norag_google.py` and `ask_rag_google.py`, that help distinguish the behavior of a RAG implementation.
- Both applications use the OpenAI SDK to interact with Google Gemini via Google's OpenAI-compatible endpoint.
- The RAG app splits the text into fixed-size chunks with an overlap (`CHUNK_SIZE`, `OVERLAP_PERCENT` and `MIN_PARAGRAPH_LENGTH` at the top of the script), indexes them into `chroma_db/` and reuses them across runs.
- There is one index per document (`DOCUMENT_FILE`) and chunking configuration. When no index exists for the current document and settings, the script asks before building one (this takes a little while).
- Debug prompt printing can be enabled by passing `debug=True` in the NoRAG app.
- RAG app prints retrieved context for verification.
- Streaming output is enabled by default in both applications (controlled by `STREAM_ON` variable).

- Using the shortened version of the text (`docs/CrimeAndPunishment-shorter.txt`)
	- Go online at https://gemini.google.com/app
	- At the prompt, start writing "Here is a story: ",
	- Then paste the content of docs/CrimeAndPunishment-shorter.txt into the Gemini chat app and then, at the end of the prompt, ask "What can you tell me about Denis Amselemovitch?"
	- Observe the response from Gemini.

- For the BM25 lexical-retrieval example, see [README_BM25.md](./README_BM25.md).

### Free books from Project Gutenberg
- https://www.gutenberg.org/
### Crime and Punishment by Fyodor Dostoevsky
- https://www.gutenberg.org/ebooks/2554
### Huckleberry Finn by Mark Twain
- https://www.gutenberg.org/ebooks/76
