"""Persistent BM25 retriever (lexical search).

Design (see DECISIONS.md, D-03 to D-06):
- Own ~100-line implementation: no extra dependency, formula visible and testable.
- Inverted index (term -> [(chunk_no, tf)]) so a query only touches chunks that
  contain one of its terms.
- Persisted as JSON in data/index/, one file per (corpus, chunk size, overlap).
  A fingerprint of the chunk texts + BM25/tokenizer parameters is stored inside:
  if documents are uploaded, deleted, (de)selected, re-chunked, or the tokenizer
  changes, the fingerprint no longer matches and the index is rebuilt
  automatically on the next query. There is no incremental update: rebuilding a
  corpus of a few dozen chunks takes milliseconds, and a full rebuild can never
  leave the index half-updated.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from typing import Any

from . import core

BM25_K1 = 1.5
BM25_B = 0.75
TOKENIZER_VERSION = "v1-fold-compound"

# Small English function-word list. Measured on our questions (see DECISIONS.md D-05):
# without it, a question full of "what / how / should / and" lets low-IDF words
# add up and outrank the chunk that holds the one rare identifier. English only:
# both corpora are English. Switch with BM25_STOPWORDS=0 or set_stopwords(False).
STOPWORDS = frozenset(
    "a an the and or of to in on at for with by from is are was were be been being "
    "it its this that these those what which who whom how do does did can could "
    "should would will i my me we our you your there their they them he she his her "
    "if as so than then when where why about into over also any some such".split()
)
_use_stopwords = os.getenv("BM25_STOPWORDS", "1") != "0"


def set_stopwords(enabled: bool) -> None:
    global _use_stopwords
    _use_stopwords = bool(enabled)


def stopwords_enabled() -> bool:
    return _use_stopwords

_WORD = re.compile(r"\w+(?:-\w+)*", re.UNICODE)


def fold(text: str) -> str:
    """Lowercase and strip accents, so 'Inès' matches 'ines'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def tokenize(text: str) -> list[str]:
    """Words, plus both the whole identifier and its parts for hyphenated ones.

    'KST-5031' -> ['kst-5031', 'kst', '5031']. The whole token is rare (high IDF)
    and pins the exact code (KST-5031 is not KST-5038); the parts keep partial
    matches working ('5031' alone, or 'KST' codes in general). Query and chunks
    go through the same function, so they always agree. No stop-word list:
    IDF already pushes frequent words to ~0.
    """
    tokens: list[str] = []

    for match in _WORD.findall(fold(text)):
        if _use_stopwords and match in STOPWORDS:
            continue

        tokens.append(match)

        if "-" in match:
            tokens.extend(part for part in match.split("-") if part)

    return tokens


class BM25Index:
    def __init__(
        self,
        ids: list[str],
        doc_len: list[int],
        postings: dict[str, list[list[int]]],
        k1: float = BM25_K1,
        b: float = BM25_B
    ):
        self.ids = ids
        self.doc_len = doc_len
        self.postings = postings
        self.k1 = k1
        self.b = b
        self.n = len(ids)
        self.avgdl = (sum(doc_len) / self.n) if self.n else 0.0

    @classmethod
    def build(cls, records: list[dict[str, Any]], k1: float = BM25_K1, b: float = BM25_B):
        postings: dict[str, dict[int, int]] = {}
        doc_len: list[int] = []

        for no, record in enumerate(records):
            tokens = tokenize(record["text"])
            doc_len.append(len(tokens))

            for token in tokens:
                counts = postings.setdefault(token, {})
                counts[no] = counts.get(no, 0) + 1

        return cls(
            [r["id"] for r in records],
            doc_len,
            {t: [[no, tf] for no, tf in c.items()] for t, c in postings.items()},
            k1,
            b
        )

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def scores(self, query: str) -> dict[int, float]:
        """Score of every chunk that shares at least one term with the query."""
        out: dict[int, float] = {}

        for term in set(tokenize(query)):
            posting = self.postings.get(term)

            if not posting:
                continue

            weight = self.idf(term)

            for no, tf in posting:
                norm = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[no] / self.avgdl
                )
                out[no] = out.get(no, 0.0) + weight * tf * (self.k1 + 1) / norm

        return out

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(
            self.scores(query).items(),
            key=lambda item: (-item[1], item[0])  # ties: earlier chunk first
        )
        return ranked[:k]

    def to_json(self) -> dict[str, Any]:
        return {
            "k1": self.k1, "b": self.b,
            "ids": self.ids, "doc_len": self.doc_len, "postings": self.postings
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]):
        return cls(data["ids"], data["doc_len"], data["postings"], data["k1"], data["b"])


class Bm25Retriever:
    """BM25 index + the chunk records it was built from."""

    def __init__(self, records: list[dict[str, Any]], index: BM25Index, info: dict[str, Any]):
        self.records = records
        self.index = index
        self.info = info
        self.backend = "bm25"

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Top-k chunks. Chunks with score 0 (no shared term) are not returned:
        they carry no lexical evidence and must not enter a fusion."""
        if not self.records:
            return []

        return [
            {**self.records[no], "score": float(score)}
            for no, score in self.index.search(query, max(1, int(k)))
            if score > 0
        ]


_CACHE: dict[str, tuple[str, Bm25Retriever]] = {}


def index_path(corpus: str, size: int, overlap: int):
    sw = "sw1" if _use_stopwords else "sw0"
    return core.INDEX_DIR / f"bm25-{core.safe_name(corpus)}-{size}-{overlap}-{sw}.json"


def get_bm25(
    corpus: str,
    size: int,
    overlap: int,
    records: list[dict[str, Any]] | None = None,
    force: bool = False
) -> Bm25Retriever:
    """Load the BM25 index from memory, else disk, else build (and persist) it.

    info["source"] says which one happened ("memory", "disk" or "built"), so
    the UI/report can show whether an update was picked up.
    """
    if records is None:
        records = core.build_chunks(corpus, size, overlap)

    fingerprint = core.records_fingerprint(
        records,
        f"bm25|{TOKENIZER_VERSION}|{BM25_K1}|{BM25_B}|stop={int(_use_stopwords)}"
    )
    path = index_path(corpus, size, overlap)
    key = str(path)

    if not force:
        cached = _CACHE.get(key)

        if cached and cached[0] == fingerprint:
            cached[1].info["source"] = "memory"
            return cached[1]

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))

                if data.get("fingerprint") == fingerprint:
                    started = time.perf_counter()
                    index = BM25Index.from_json(data["index"])
                    retriever = Bm25Retriever(records, index, {
                        "source": "disk",
                        "seconds": time.perf_counter() - started,
                        "chunks": index.n,
                        "terms": len(index.postings),
                        "path": str(path),
                        "fingerprint": fingerprint
                    })
                    _CACHE[key] = (fingerprint, retriever)
                    return retriever
            except (OSError, ValueError, KeyError, TypeError):
                pass  # unreadable or malformed: rebuild below

    started = time.perf_counter()
    index = BM25Index.build(records)
    seconds = time.perf_counter() - started

    try:
        core.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps({
                "fingerprint": fingerprint,
                "tokenizer": TOKENIZER_VERSION,
                "stopwords": _use_stopwords,
                "corpus": corpus,
                "chunk_size": size,
                "overlap": overlap,
                "index": index.to_json()
            }, ensure_ascii=False),
            encoding="utf-8"
        )
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written index
    except OSError:
        pass  # persistence is best effort; the in-memory index still works

    retriever = Bm25Retriever(records, index, {
        "source": "built",
        "seconds": seconds,
        "chunks": index.n,
        "terms": len(index.postings),
        "path": str(path),
        "fingerprint": fingerprint
    })
    _CACHE[key] = (fingerprint, retriever)

    return retriever
