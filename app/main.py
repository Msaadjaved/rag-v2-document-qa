
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables before importing core.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from .core import (
    DATA,
    build_chunks,
    read_corpus,
    load_questions,
    Retriever,
    generate_answer,
    words_for_corpus,
    question_words,
    safe_name,
    save_json
)

app = FastAPI(title="RAG V3", version="3.0.0")

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "app" / "static"),
    name="static"
)


class AskRequest(BaseModel):
    corpus: Literal["kestrel", "own"] = "own"
    question: str = Field(min_length=2, max_length=4000)
    chunk_size: int = Field(default=800, ge=100, le=10000)
    overlap: int = Field(default=10, ge=0, le=89)
    top_k: int = Field(default=5, ge=1, le=20)


class EvalRequest(BaseModel):
    corpus: Literal["kestrel", "own"] = "own"
    chunk_size: int = Field(default=800, ge=100, le=10000)
    overlap: int = Field(default=10, ge=0, le=89)
    top_k: int = Field(default=5, ge=1, le=20)
    generate_answers: bool = False


@app.get("/", response_class=HTMLResponse)
def home():
    return (
        ROOT / "app" / "static" / "index.html"
    ).read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    return {
        "corpora": {
            corpus: {
                "documents": len(read_corpus(corpus)),
                "chunks": len(build_chunks(corpus, 800, 10)),
                "questions": len(load_questions(corpus))
            }
            for corpus in ["kestrel", "own"]
        },
        "llm_configured": bool(
            os.getenv("LLM_API_KEY")
            and os.getenv("LLM_BASE_URL")
            and os.getenv("LLM_MODEL")
        ),
        "deadline_note": (
            "Kestrel course corpus and official starter questions are installed."
        )
    }


@app.get("/api/documents")
def documents(corpus: Literal["kestrel", "own"] = "own"):
    folder = DATA / "corpora" / corpus

    all_files = sorted([
        *folder.glob("*.txt"),
        *folder.glob("*.md")
    ]) if folder.exists() else []

    selection_file = folder / "selection.json"

    selected = (
        set(json.loads(selection_file.read_text(encoding="utf-8")))
        if selection_file.exists()
        else {p.name for p in all_files}
    )

    return [
        {
            "name": p.name,
            "characters": len(p.read_text(encoding="utf-8")),
            "selected": p.name in selected
        }
        for p in all_files
    ]


class SelectionRequest(BaseModel):
    selected: list[str]


@app.put("/api/documents/{corpus}/selection")
def update_selection(
    corpus: Literal["kestrel", "own"],
    req: SelectionRequest
):
    folder = DATA / "corpora" / corpus

    allowed = {
        p.name for p in [
            *folder.glob("*.txt"),
            *folder.glob("*.md")
        ]
    } if folder.exists() else set()

    if any(name not in allowed for name in req.selected):
        raise HTTPException(
            400,
            "Selection contains an unknown document"
        )

    save_json(folder / "selection.json", sorted(set(req.selected)))

    return {
        "selected": sorted(set(req.selected)),
        "available": len(allowed)
    }


@app.post("/api/documents/{corpus}")
async def upload_document(
    corpus: Literal["kestrel", "own"],
    file: UploadFile = File(...)
):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in {".txt", ".md"}:
        raise HTTPException(
            400,
            "Only UTF-8 .txt and .md documents are supported."
        )

    folder = DATA / "corpora" / corpus
    folder.mkdir(parents=True, exist_ok=True)

    name = safe_name(Path(file.filename or "").stem) + suffix

    content = await file.read()

    if len(content) > 20_000_000:
        raise HTTPException(413, "File exceeds 20 MB.")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            400,
            "Document must be UTF-8 encoded."
        )

    (folder / name).write_text(text, encoding="utf-8")

    return {
        "uploaded": name,
        "characters": len(text)
    }


@app.delete("/api/documents/{corpus}/{name}")
def delete_document(
    corpus: Literal["kestrel", "own"],
    name: str
):
    original = Path(name)
    suffix = original.suffix.lower()

    if suffix not in {".txt", ".md"}:
        raise HTTPException(400, "Unsupported document type")

    path = (
        DATA
        / "corpora"
        / corpus
        / (safe_name(original.stem) + suffix)
    )

    if not path.exists():
        raise HTTPException(404, "Document not found")

    path.unlink()

    return {"deleted": path.name}


@app.post("/api/ask")
def ask(req: AskRequest):
    records = build_chunks(
        req.corpus,
        req.chunk_size,
        req.overlap
    )

    if not records:
        raise HTTPException(
            400,
            f"No documents found for {req.corpus}. Upload the corpus documents first."
        )

    retriever = Retriever(records)
    hits = retriever.search(req.question, req.top_k)

    try:
        raw = generate_answer(req.question, hits)
        answer = json.loads(raw)

    except Exception as e:
        raise HTTPException(503, str(e))

    return {
        "answer": answer["answer"],
        "citations": [
            {**hit, "citation": hit["id"]}
            for hit in hits
            if hit["id"] in answer["citations"]
        ],
        "retrieved": hits,
        "backend": retriever.backend
    }


@app.get("/api/words/{corpus}")
def words(
    corpus: Literal["kestrel", "own"],
    chunk_size: int = 800,
    overlap: int = 10,
    limit: int = 300
):
    result = words_for_corpus(corpus, chunk_size, overlap)

    result["vocabulary"] = result["vocabulary"][
        :max(1, min(limit, 2000))
    ]

    return result


@app.get("/api/question-words/{corpus}")
def qwords(
    corpus: Literal["kestrel", "own"],
    question: str,
    chunk_size: int = 800,
    overlap: int = 10
):
    return question_words(
        corpus,
        question,
        chunk_size,
        overlap
    )


@app.get("/api/questions/{corpus}")
def questions(corpus: Literal["kestrel", "own"]):
    return load_questions(corpus)


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    questions = load_questions(req.corpus)

    records = build_chunks(
        req.corpus,
        req.chunk_size,
        req.overlap
    )

    if not records:
        raise HTTPException(
            400,
            "No corpus documents found. Upload the corpus first."
        )

    if not questions:
        raise HTTPException(
            400,
            f"No question set found at data/questions/{req.corpus}.jsonl"
        )

    retriever = Retriever(records)
    rows = []

    for q in questions:

        # Retrieve once, with enough results for all evaluation cutoffs.
        all_hits = retriever.search(
            q["question"],
            max(req.top_k, 10)
        )

        # Keep configured top_k for answer generation and stored retrieval.
        hits = all_hits[:req.top_k]

        evidence = q.get("evidence", [])

        quotes = [
            str(item.get("quote", "")).strip().lower()
            for item in evidence
            if isinstance(item, dict)
        ]

        legacy_quote = str(
            q.get("answer_quote", "")
        ).strip().lower()

        if legacy_quote:
            quotes.append(legacy_quote)

        relevant = []

        for record in records:
            matching_quotes = [
                quote for quote in quotes
                if quote and quote in record["text"].lower()
            ]

            matching_sources = {
                str(item.get("source", ""))
                for item in evidence
                if (
                    isinstance(item, dict)
                    and str(item.get("quote", "")).strip().lower()
                    in matching_quotes
                )
            }

            if matching_quotes and (
                not matching_sources
                or record["document"] in matching_sources
            ):
                relevant.append(record)

        relevant_ids = {r["id"] for r in relevant}

        # Configured top_k retrieval hit.
        retrieved_ids = {r["id"] for r in hits}

        row = {
            "id": q.get("id"),
            "question": q["question"],
            "type": q.get("type", "unspecified"),
            "answerable": bool(q.get("answerable", True)),
            "relevant_chunk_ids": sorted(relevant_ids),
            "retrieved": [
                {
                    k: r[k]
                    for k in ["id", "document", "index", "score", "text"]
                }
                for r in hits
            ],
            "hit_at_k": (
                bool(relevant_ids & retrieved_ids)
                if q.get("answerable", True)
                else None
            )
        }

        # Calculate all cutoffs from the SAME ranked retrieval result.
        # This guarantees nested and consistent Hit@k values.
        for n in (1, 3, 5, 10):
            top_ids = {
                r["id"] for r in all_hits[:n]
            }

            row[f"hit_at_{n}"] = (
                bool(relevant_ids & top_ids)
                if q.get("answerable", True)
                else None
            )

        if req.generate_answers:
            try:
                row["generated_answer"] = json.loads(
                    generate_answer(q["question"], hits)
                )
            except Exception as e:
                row["generation_error"] = str(e)

        rows.append(row)

    answerable = [
        r for r in rows if r["answerable"]
    ]

    hit_rates = {}

    for n in (1, 3, 5, 10):
        hit_rates[f"hit_rate_at_{n}"] = (
            sum(r[f"hit_at_{n}"] for r in answerable)
            / len(answerable)
            if answerable else None
        )

    by_type = {}

    for question_type in sorted({
        r["type"] for r in answerable
    }):
        typed_rows = [
            r for r in answerable
            if r["type"] == question_type
        ]

        by_type[question_type] = {
            "count": len(typed_rows),
            **{
                f"hit_rate_at_{n}": (
                    sum(r[f"hit_at_{n}"] for r in typed_rows)
                    / len(typed_rows)
                )
                for n in (1, 3, 5, 10)
            }
        }

    summary = {
        "corpus": req.corpus,
        "config": {
            "chunk_size": req.chunk_size,
            "overlap": req.overlap,
            "top_k": req.top_k,
            "retriever": retriever.backend
        },
        "total_questions": len(rows),
        "answerable_questions": len(answerable),
        "hit_rates": hit_rates,
        "hit_rate_at_k": (
            sum(r["hit_at_k"] for r in answerable) / len(answerable)
            if answerable else None
        ),
        "by_type": by_type
    }

    stamp = (
        time.strftime("%Y%m%d-%H%M%S")
        + f"-{time.time_ns() % 1_000_000_000:09d}"
    )

    path = DATA / "results" / f"{req.corpus}-{stamp}.json"

    save_json(path, {
        "summary": summary,
        "results": rows
    })

    return {
        "summary": summary,
        "results": rows,
        "saved_to": str(path.relative_to(ROOT))
    }


@app.post("/api/no-context")
def no_context(req: EvalRequest):
    """Run the question set against the configured LLM without retrieved context."""

    questions = load_questions(req.corpus)

    if not questions:
        raise HTTPException(
            400,
            f"No question set found at data/questions/{req.corpus}.jsonl"
        )

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if not api_key or not base_url or not model:
        raise HTTPException(
            503,
            "LLM configuration is incomplete. "
            "Check LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env."
        )

    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    rows = []

    for q in questions:
        prompt = (
            "Answer using your own knowledge only, without documents or tools. "
            "If unsure, say so.\n\nQUESTION: "
            + q["question"]
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048
            )

            rows.append({
                "id": q.get("id"),
                "question": q["question"],
                "type": q.get("type"),
                "answer": response.choices[0].message.content or ""
            })

        except Exception as e:
            rows.append({
                "id": q.get("id"),
                "question": q["question"],
                "error": str(e)
            })

    path = DATA / "results" / (
        f"{req.corpus}-no-context-"
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{time.time_ns() % 1_000_000_000:09d}.json"
    )

    save_json(path, {
        "corpus": req.corpus,
        "mode": "no-context",
        "provider_base_url": base_url,
        "model": model,
        "results": rows
    })

    return {
        "results": rows,
        "saved_to": str(path.relative_to(ROOT))
    }


@app.get("/api/results")
def results():
    folder = DATA / "results"

    return [
        {
            "file": p.name,
            "path": str(p.relative_to(ROOT))
        }
        for p in sorted(folder.glob("*.json"))
    ] if folder.exists() else []
