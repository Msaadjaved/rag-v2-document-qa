"""Reciprocal Rank Fusion (RRF).

score(chunk) = sum over retrievers r that returned it of  weight_r / (k + rank_r)

Only ranks are used, never raw scores, because cosine similarities, BM25 scores
and graph scores live on different scales. k=60 is the value of the original RRF
paper (Cormack et al., 2009); a smaller k favours the very top of each list.
"""
from __future__ import annotations

from typing import Any


def rrf(
    ranked_lists: dict[str, list[dict[str, Any]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    limit: int | None = None
) -> list[dict[str, Any]]:
    """Fuse ranked lists (best first) into one list.

    Each returned chunk has "score" (the RRF score) and "sources":
    {retriever: {"rank": 1-based rank in that retriever, "score": its raw score}},
    which answers "which retriever(s) found it, and at which rank?".
    """
    weights = weights or {}
    table: dict[str, dict[str, Any]] = {}

    for name, hits in ranked_lists.items():
        weight = float(weights.get(name, 1.0))

        for rank, hit in enumerate(hits, start=1):
            entry = table.setdefault(hit["id"], {
                "record": {key: v for key, v in hit.items() if key != "score"},
                "score": 0.0,
                "sources": {}
            })
            entry["score"] += weight / (k + rank)
            entry["sources"][name] = {
                "rank": rank,
                "score": float(hit.get("score", 0.0))
            }

    ordered = sorted(
        table.values(),
        key=lambda e: (
            -e["score"],
            min(s["rank"] for s in e["sources"].values()),  # tie: best single rank
            e["record"]["id"]                                # then deterministic
        )
    )

    fused = [
        {**e["record"], "score": e["score"], "sources": e["sources"]}
        for e in ordered
    ]

    return fused[:limit] if limit else fused
