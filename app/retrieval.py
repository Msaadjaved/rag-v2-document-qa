"""Retriever orchestration: switch retrievers on/off, fuse with RRF, report costs.

Used by the API (/api/ask, /api/evaluate) and by the CLI evaluator, so a
configuration means exactly the same thing everywhere.

Graph retriever contract (implemented by app/graph_retriever.py, owned by the
concept-graph branch). It must expose:

    search(corpus: str, query: str, k: int, chunk_size: int, overlap: int) -> dict

returning
    {
      "hits":  [{"id": <chunk id as built by core.build_chunks>, "score": float}, ...],  # best first
      "path":  {"entities": [...], "edges": [...], "chunks": [...]},                      # shown in the UI
      "usage": {"llm_calls": int, "prompt_tokens": int, "completion_tokens": int}         # query-time cost (may be 0)
    }
Chunk ids depend on chunk size and overlap: the graph's provenance must use the
same chunking as the other retrievers.
"""
from __future__ import annotations

import os
import time
from typing import Any

from . import bm25 as bm25_module
from . import core
from .fusion import rrf

RETRIEVERS = ("dense", "bm25", "graph")
GRAPH_CORPORA = {"kestrel"}


class RetrieverUnavailable(RuntimeError):
    """A requested retriever cannot run (e.g. the graph is not built yet)."""


def default_retrievers() -> list[str]:
    raw = os.getenv("DEFAULT_RETRIEVERS", "dense,bm25")
    chosen = [r.strip() for r in raw.split(",") if r.strip() in RETRIEVERS]
    return chosen or ["dense", "bm25"]


def graph_available() -> bool:
    try:
        from . import graph_retriever  # noqa: F401
        return True
    except ImportError:
        return False


def normalize_retrievers(corpus: str, retrievers: list[str]) -> list[str]:
    names = list(dict.fromkeys(retrievers))  # de-duplicate, keep order

    if not names:
        raise ValueError("Switch on at least one retriever.")

    unknown = [n for n in names if n not in RETRIEVERS]

    if unknown:
        raise ValueError(f"Unknown retriever(s): {unknown}. Choose from {list(RETRIEVERS)}.")

    if "graph" in names and corpus not in GRAPH_CORPORA:
        raise ValueError("The concept graph is only built for the Kestrel corpus.")

    return names


def _run_graph(corpus: str, question: str, n: int, size: int, overlap: int) -> dict[str, Any]:
    try:
        from . import graph_retriever
    except ImportError as exc:
        raise RetrieverUnavailable(
            "Graph retriever not available: app/graph_retriever.py is missing."
        ) from exc

    return graph_retriever.search(
        corpus=corpus, query=question, k=n, chunk_size=size, overlap=overlap
    )


def retrieve(
    corpus: str,
    question: str,
    size: int,
    overlap: int,
    retrievers: list[str],
    depth: int = 5,
    candidates: int = 10,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
    strict_dense: bool = False,
    records: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Run the retrievers that are on and return the top `depth` chunks.

    One retriever on: its own ranking (no fusion). Several on: RRF over their
    top `candidates` lists. Every returned chunk carries "sources".
    """
    names = normalize_retrievers(corpus, retrievers)

    if records is None:
        records = core.build_chunks(corpus, size, overlap)

    by_id = {r["id"]: r for r in records}
    n = max(int(candidates), int(depth))

    lists: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, float] = {}
    backend: dict[str, Any] = {}
    usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    graph_path = None
    started = time.perf_counter()

    for name in names:
        t0 = time.perf_counter()

        if name == "dense":
            dense = core.get_dense(corpus, size, overlap, records=records, strict=strict_dense)
            lists["dense"] = dense.search(question, n)
            backend["dense"] = dense.backend

        elif name == "bm25":
            lexical = bm25_module.get_bm25(corpus, size, overlap, records=records)
            lists["bm25"] = lexical.search(question, n)
            backend["bm25"] = {
                "index": lexical.info.get("source"),
                "stopwords": bm25_module.stopwords_enabled()
            }

        else:  # graph
            out = _run_graph(corpus, question, n, size, overlap)
            graph_path = out.get("path")
            lists["graph"] = [
                {**by_id[h["id"]], "score": float(h.get("score", 0.0))}
                for h in out.get("hits", [])
                if h.get("id") in by_id  # drop ids that are not in this chunking
            ]

            for key in usage:
                usage[key] += int((out.get("usage") or {}).get(key, 0) or 0)

        timings[f"{name}_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()

    if len(lists) == 1:
        (only, hits), = lists.items()
        fused = [
            {**{k: v for k, v in h.items() if k != "score"},
             "score": float(h["score"]),
             "sources": {only: {"rank": i, "score": float(h["score"])}}}
            for i, h in enumerate(hits[:depth], start=1)
        ]
    else:
        fused = rrf(lists, k=rrf_k, weights=weights, limit=depth)

    timings["fusion_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - started

    return {
        "hits": fused,
        "retrievers": names,
        "graph_path": graph_path,
        "timings": timings,
        "usage": usage,
        "backend": backend,
        "rrf_k": rrf_k,
        "candidates": n
    }
