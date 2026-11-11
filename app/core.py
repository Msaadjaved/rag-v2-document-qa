"""Core RAG, corpus, words analytics and evaluation logic for RAG V2."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
INDEX_DIR = DATA / "index"  # derived data (BM25 index, embeddings): git-ignored

load_dotenv(ROOT / ".env")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
OVERLAP = int(os.getenv("OVERLAP", "10"))


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return value[:60] or "document"


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap_percent: int = OVERLAP
) -> list[str]:
    if size < 100 or not 0 <= overlap_percent < 90:
        raise ValueError(
            "Chunk size must be >=100 and overlap must be between 0 and 89%"
        )

    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()
    ]
    normalized = "\n\n".join(paragraphs)

    step = max(1, size - size * overlap_percent // 100)

    return [
        normalized[i:i + size]
        for i in range(0, len(normalized), step)
        if normalized[i:i + size].strip()
    ]


def read_corpus(corpus: str) -> list[dict[str, Any]]:
    folder = DATA / "corpora" / safe_name(corpus)

    files = sorted([
        *folder.glob("*.txt"),
        *folder.glob("*.md")
    ]) if folder.exists() else []

    selection_file = folder / "selection.json"

    if files and selection_file.exists():
        try:
            selected = set(
                json.loads(selection_file.read_text(encoding="utf-8"))
            )
            files = [path for path in files if path.name in selected]
        except (OSError, ValueError, TypeError):
            pass

    # Default own corpus: Vellora espresso machine manual.
    if corpus == "own" and not files:
        files = [DOCS / "vellora-duo-espresso-machine.txt"]

    docs = []

    for path in files:
        if path.is_file():
            docs.append({
                "name": path.name,
                "text": path.read_text(encoding="utf-8")
            })

    return docs


def build_chunks(
    corpus: str,
    size: int,
    overlap: int
) -> list[dict[str, Any]]:
    records = []

    for doc in read_corpus(corpus):
        for idx, chunk in enumerate(
            chunk_text(doc["text"], size, overlap)
        ):
            records.append({
                "id": f"{safe_name(doc['name'])}:{idx}",
                "document": doc["name"],
                "index": idx,
                "text": chunk
            })

    return records


def records_fingerprint(records: list[dict[str, Any]], extra: str = "") -> str:
    """Stable hash of the chunk list (ids + texts, in order) and of any extra
    parameters. Used to know when a persisted index is stale: upload, delete,
    selection change, new chunk size or new overlap all change the fingerprint."""
    h = hashlib.sha256()
    h.update(extra.encode("utf-8"))
    h.update(b"\x02")
    for r in records:
        h.update(r["id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(r["text"].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None


def get_embedding_model():
    """Load the sentence-transformers model once per process."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(DENSE_MODEL_NAME)
    return _MODEL


class Retriever:
    """Dense retriever (SentenceTransformers), with a TF-IDF fallback.

    strict=True raises instead of silently falling back to TF-IDF: use it for
    every evaluation, otherwise "dense" numbers may really be TF-IDF numbers.
    cache_key enables an on-disk embedding cache (data/index/dense-<key>.npz),
    validated by a fingerprint of the chunks.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        backend: str = "auto",
        strict: bool = False,
        cache_key: str | None = None
    ):
        self.records = records
        self.backend = "tfidf"
        self.model = None
        self.matrix = None
        self.cache_hit = False
        self.build_seconds = 0.0
        started = time.perf_counter()

        if records and backend != "tfidf":
            try:
                self.model = get_embedding_model()
                self.matrix, self.cache_hit = self._embeddings(cache_key)
                self.backend = "sentence-transformers"

            except Exception as exc:
                self.model = None
                self.matrix = None

                if strict:
                    raise RuntimeError(
                        "Dense backend unavailable (sentence-transformers "
                        f"failed: {exc}). Refusing the TF-IDF fallback in "
                        "strict mode."
                    ) from exc

        if self.backend == "tfidf" and records:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )

            self.matrix = self.vectorizer.fit_transform(
                [r["text"] for r in records]
            )

            self._cosine = cosine_similarity

        self.build_seconds = time.perf_counter() - started
        print(
            f"Retriever backend selected: {self.backend} "
            f"(embedding cache hit: {self.cache_hit}, "
            f"{self.build_seconds:.2f}s)"
        )

    def _embeddings(self, cache_key: str | None):
        import numpy as np

        fingerprint = records_fingerprint(self.records, DENSE_MODEL_NAME)
        path = INDEX_DIR / f"dense-{cache_key}.npz" if cache_key else None

        if path is not None and path.exists():
            try:
                with np.load(path, allow_pickle=False) as z:
                    if str(z["fingerprint"]) == fingerprint:
                        return z["matrix"], True
            except Exception:
                pass  # unreadable cache: just recompute

        matrix = self.model.encode(
            [r["text"] for r in self.records],
            normalize_embeddings=True
        )

        if path is not None:
            try:
                INDEX_DIR.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp.npz")
                np.savez(tmp, matrix=matrix, fingerprint=np.array(fingerprint))
                os.replace(tmp, path)
            except OSError:
                pass  # cache is an optimisation, never a failure

        return matrix, False

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.records:
            return []

        k = max(1, min(int(k), len(self.records)))

        if self.backend == "sentence-transformers":
            q = self.model.encode(
                [query],
                normalize_embeddings=True
            )[0]

            scores = self.matrix @ q

        else:
            q = self.vectorizer.transform([query])
            scores = self._cosine(q, self.matrix)[0]

        indexes = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True
        )[:k]

        return [
            {**self.records[i], "score": float(scores[i])}
            for i in indexes
        ]


_DENSE_CACHE: dict[tuple, tuple[str, Retriever]] = {}


def get_dense(
    corpus: str,
    size: int,
    overlap: int,
    records: list[dict[str, Any]] | None = None,
    strict: bool = False
) -> Retriever:
    """Return a cached dense retriever, rebuilt only if the chunks changed."""
    if records is None:
        records = build_chunks(corpus, size, overlap)

    fingerprint = records_fingerprint(records, DENSE_MODEL_NAME)
    key = (corpus, size, overlap)
    cached = _DENSE_CACHE.get(key)

    if cached and cached[0] == fingerprint and (
        cached[1].backend == "sentence-transformers" or not strict
    ):
        return cached[1]

    retriever = Retriever(
        records,
        strict=strict,
        cache_key=f"{safe_name(corpus)}-{size}-{overlap}"
    )
    _DENSE_CACHE[key] = (fingerprint, retriever)

    return retriever


def words_for_corpus(
    corpus: str,
    size: int,
    overlap: int
) -> dict[str, Any]:
    records = build_chunks(corpus, size, overlap)
    vocab: dict[str, dict[str, Any]] = {}

    for record in records:
        counts: dict[str, int] = {}

        for word in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9'-]*",
            record["text"].lower()
        ):
            counts[word] = counts.get(word, 0) + 1

        for word, count in counts.items():
            item = vocab.setdefault(word, {
                "word": word,
                "occurrences": 0,
                "chunk_count": 0,
                "chunks": []
            })

            item["occurrences"] += count
            item["chunk_count"] += 1

            item["chunks"].append({
                "id": record["id"],
                "document": record["document"],
                "count": count,
                "text": record["text"]
            })

    return {
        "chunk_count": len(records),
        "vocabulary": sorted(
            vocab.values(),
            key=lambda x: (-x["occurrences"], x["word"])
        )
    }


def question_words(
    corpus: str,
    question: str,
    size: int,
    overlap: int
) -> dict[str, Any]:
    vocab = words_for_corpus(corpus, size, overlap)["vocabulary"]

    index = {v["word"]: v for v in vocab}

    words = list(dict.fromkeys(
        re.findall(
            r"[A-Za-z0-9][A-Za-z0-9'-]*",
            question.lower()
        )
    ))

    return {
        "present": [
            {"word": word, "chunks": index[word]["chunks"]}
            for word in words if word in index
        ],
        "absent": [
            word for word in words if word not in index
        ]
    }


def load_questions(corpus: str) -> list[dict[str, Any]]:
    path = DATA / "questions" / f"{safe_name(corpus)}.jsonl"

    if not path.exists():
        return []

    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    return rows


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def generate_answer(
    question: str,
    retrieved: list[dict[str, Any]]
) -> str:
    """Generate a grounded answer using the configured OpenAI-compatible API."""
    return generate_answer_with_usage(question, retrieved)[0]


def generate_answer_with_usage(
    question: str,
    retrieved: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    """Same as generate_answer, plus {"llm_calls", "prompt_tokens",
    "completion_tokens", "latency_s"} for cost reporting."""

    load_dotenv(ROOT / ".env")

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if not api_key or not base_url or not model:
        raise RuntimeError(
            "LLM configuration is incomplete. Set LLM_BASE_URL, "
            "LLM_API_KEY and LLM_MODEL in your .env file."
        )

    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    context = "\n\n".join(
        f"[{r['id']}] {r['text']}"
        for r in retrieved
    )

    prompt = (
        "Answer using only the supplied document chunks. "
        "If they do not contain the answer, say that the information "
        "is not in the documents. Return valid JSON with exactly two keys: "
        '"answer" (a string) and "citations" (an array of chunk IDs). '
        "Only cite IDs that appear in the supplied chunks.\n\n"
        "CHUNKS:\n"
        + context
        + "\n\nQUESTION:\n"
        + question
    )

    started = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048
    )

    usage_obj = getattr(response, "usage", None)

    usage = {
        "llm_calls": 1,
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(
            getattr(usage_obj, "completion_tokens", 0) or 0
        ),
        "latency_s": time.perf_counter() - started
    }

    content = response.choices[0].message.content or "{}"

    try:
        data = json.loads(content)

        if (
            not isinstance(data.get("answer"), str)
            or not isinstance(data.get("citations"), list)
        ):
            raise ValueError("Invalid LLM response structure")

        allowed = {r["id"] for r in retrieved}

        data["citations"] = [
            citation
            for citation in data["citations"]
            if isinstance(citation, str) and citation in allowed
        ]

        return json.dumps(data, ensure_ascii=False), usage

    except (json.JSONDecodeError, ValueError):
        return json.dumps({
            "answer": content,
            "citations": []
        }, ensure_ascii=False), usage
