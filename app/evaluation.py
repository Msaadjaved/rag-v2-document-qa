"""Retrieval evaluation: HitRate@k and Recall@k, overall and per question type.

Definitions (also in README / REPORT):
- A question has "hops": one per evidence quote (two_hop questions have several;
  a legacy `answer_quote` is one hop). The gold chunks of a hop are the chunks
  whose text contains the quote (case-insensitive, same document if `source` is
  given). Same matching rule as V2, so V3 numbers stay comparable to V2.
- Hit@k    = 1 if at least one hop has a gold chunk in the top k.   (V2's metric)
- Recall@k = (hops with a gold chunk in the top k) / (all hops).     (new)
  Recall tells whether ALL the hops of a chain were retrieved. A hop whose quote
  matches no chunk (quote cut by a chunk boundary) counts as not retrieved and is
  listed in `unmatched_hops`: it is an annotation/chunking problem, not noise.
- not_in_corpus questions (answerable=false) have no gold chunk: they are
  excluded from both metrics, as in V2, but still run (for cost and for display).
- Cost per question: retrieval latency per retriever, LLM calls, prompt and
  completion tokens, generation latency (only when generate_answers=True).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from . import bm25 as bm25_module
from . import core
from .retrieval import retrieve

CUTOFFS = (1, 3, 5, 10)

# V2 question files use a few different spellings. We map them when reporting,
# and keep the original in `type_raw`: the question files are never edited.
# `not_in_corpus_trap` (OWN-04) is deliberately NOT mapped to `not_in_corpus`:
# that question is answerable (the manual says "never use vinegar").
TYPE_ALIASES = {
    "different_vocabulary": "vocabulary",
    "two_hops": "two_hop"
}

TYPE_ORDER = [
    "exact_identifier", "split_answer", "vocabulary",
    "not_in_corpus", "versions", "two_hop"
]


def canonical_type(raw: str | None) -> str:
    raw = raw or "unspecified"
    return TYPE_ALIASES.get(raw, raw)


def question_hops(question: dict[str, Any]) -> list[dict[str, Any]]:
    hops: list[dict[str, Any]] = []

    for item in question.get("evidence") or []:
        if isinstance(item, dict) and str(item.get("quote", "")).strip():
            hops.append({
                "source": item.get("source") or None,
                "quote": str(item["quote"]).strip()
            })

    legacy = str(question.get("answer_quote", "")).strip()

    if legacy and all(h["quote"].lower() != legacy.lower() for h in hops):
        hops.append({"source": None, "quote": legacy})

    return hops


def gold_chunks(records: list[dict[str, Any]], hop: dict[str, Any]) -> list[str]:
    quote = hop["quote"].lower()

    return [
        r["id"] for r in records
        if quote in r["text"].lower()
        and (not hop["source"] or r["document"] == hop["source"])
    ]


def score_ranking(
    hop_gold: list[set[str]],
    ranked_ids: list[str],
    cutoffs: tuple[int, ...]
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for n in cutoffs:
        top = set(ranked_ids[:n])
        covered = sum(1 for gold in hop_gold if gold & top)
        out[f"hit_at_{n}"] = covered > 0
        out[f"recall_at_{n}"] = (covered / len(hop_gold)) if hop_gold else None

    return out


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]

    return (sum(values) / len(values)) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        **{
            f"hit_rate_at_{n}": _mean([float(r[f"hit_at_{n}"]) for r in rows])
            for n in CUTOFFS
        },
        **{
            f"recall_at_{n}": _mean([r[f"recall_at_{n}"] for r in rows])
            for n in CUTOFFS
        }
    }


def _warm(
    corpus: str,
    size: int,
    overlap: int,
    names: list[str],
    records: list[dict[str, Any]],
    strict_dense: bool
) -> dict[str, Any]:
    """Build/load the indexes once, before timing any question, and report
    what that cost (so per-question latency is steady-state latency)."""
    info: dict[str, Any] = {}

    if "dense" in names:
        dense = core.get_dense(corpus, size, overlap, records=records, strict=strict_dense)
        info["dense"] = {
            "backend": dense.backend,
            "build_seconds": dense.build_seconds,
            "embedding_cache_hit": dense.cache_hit
        }

    if "bm25" in names:
        lexical = bm25_module.get_bm25(corpus, size, overlap, records=records)
        info["bm25"] = {
            "source": lexical.info.get("source"),
            "build_seconds": lexical.info.get("seconds"),
            "terms": lexical.info.get("terms"),
            "stopwords": bm25_module.stopwords_enabled()
        }

    return info


def evaluate_corpus(
    corpus: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
    retrievers: list[str],
    generate_answers: bool = False,
    rrf_k: int = 60,
    candidates: int = 10,
    weights: dict[str, float] | None = None,
    strict_dense: bool = False,
    label: str | None = None,
    questions: list[dict[str, Any]] | None = None,
    records: list[dict[str, Any]] | None = None,
    progress: Callable[[int, int, str], None] | None = None
) -> dict[str, Any]:
    questions = core.load_questions(corpus) if questions is None else questions
    records = core.build_chunks(corpus, chunk_size, overlap) if records is None else records

    if not records:
        raise ValueError("No corpus documents found. Upload the corpus first.")

    if not questions:
        raise ValueError(f"No question set found at data/questions/{corpus}.jsonl")

    depth = max(top_k, max(CUTOFFS))
    cutoffs = tuple(sorted(set(CUTOFFS) | {top_k}))
    index_info = _warm(corpus, chunk_size, overlap, list(dict.fromkeys(retrievers)), records, strict_dense)
    llm_ready = bool(
        os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL")
    )

    rows: list[dict[str, Any]] = []

    for number, q in enumerate(questions, start=1):
        if progress:
            progress(number, len(questions), str(q.get("id")))

        answerable = bool(q.get("answerable", True))
        hops = question_hops(q)

        for hop in hops:
            hop["gold_chunk_ids"] = gold_chunks(records, hop)

        out = retrieve(
            corpus, q["question"], chunk_size, overlap, retrievers,
            depth=depth, candidates=candidates, rrf_k=rrf_k,
            weights=weights, strict_dense=strict_dense, records=records
        )

        ranked = [h["id"] for h in out["hits"]]
        hop_gold = [set(h["gold_chunk_ids"]) for h in hops]
        scored = score_ranking(hop_gold, ranked, cutoffs)
        all_gold = set().union(*hop_gold) if hop_gold else set()
        first_gold_rank = next(
            (i for i, cid in enumerate(ranked, start=1) if cid in all_gold), None
        )

        by_id = {h["id"]: h for h in out["hits"]}

        for hop in hops:
            hop["retrieved_by"] = {}  # retriever -> best rank, among the gold chunks it found

            for cid in hop["gold_chunk_ids"]:
                for name, src in (by_id.get(cid, {}).get("sources") or {}).items():
                    best = hop["retrieved_by"].get(name)
                    hop["retrieved_by"][name] = src["rank"] if best is None else min(best, src["rank"])

            hop["found_in_top_k"] = any(cid in set(ranked[:top_k]) for cid in hop["gold_chunk_ids"])

        hits = out["hits"][:top_k]
        usage = dict(out["usage"])
        generation_latency = 0.0

        row: dict[str, Any] = {
            "id": q.get("id"),
            "question": q["question"],
            "type": canonical_type(q.get("type")),
            "type_raw": q.get("type"),
            "answerable": answerable,
            "relevant_chunk_ids": sorted(all_gold),
            "first_gold_rank": first_gold_rank if answerable else None,
            "hops": hops,
            "unmatched_hops": [h["quote"] for h in hops if not h["gold_chunk_ids"]],
            "retrieved": [
                {k: h[k] for k in ("id", "document", "index", "score", "text", "sources")}
                for h in hits
            ],
            "hit_at_k": scored[f"hit_at_{top_k}"] if answerable else None,
            "recall_at_k": scored[f"recall_at_{top_k}"] if answerable else None,
            "graph_path": out["graph_path"]
        }

        for n in cutoffs:
            row[f"hit_at_{n}"] = scored[f"hit_at_{n}"] if answerable else None
            row[f"recall_at_{n}"] = scored[f"recall_at_{n}"] if answerable else None

        if generate_answers:
            if not llm_ready:
                row["generation_error"] = "LLM_API_KEY / LLM_BASE_URL / LLM_MODEL not set"
            else:
                try:
                    raw, gen = core.generate_answer_with_usage(q["question"], hits)
                    row["generated_answer"] = json.loads(raw)
                    usage["llm_calls"] += gen["llm_calls"]
                    usage["prompt_tokens"] += gen["prompt_tokens"]
                    usage["completion_tokens"] += gen["completion_tokens"]
                    generation_latency = gen["latency_s"]
                except Exception as exc:  # keep evaluating the other questions
                    row["generation_error"] = str(exc)

        row["cost"] = {
            "retrieval_s": out["timings"]["total_s"],
            "per_retriever_s": {
                k: v for k, v in out["timings"].items() if k.endswith("_s") and k != "total_s"
            },
            "generation_s": generation_latency,
            "latency_s": out["timings"]["total_s"] + generation_latency,
            **usage
        }

        rows.append(row)

    answerable_rows = [r for r in rows if r["answerable"]]
    types = sorted({r["type"] for r in rows}, key=lambda t: (TYPE_ORDER.index(t) if t in TYPE_ORDER else 99, t))

    by_type: dict[str, Any] = {}

    for t in types:
        typed = [r for r in rows if r["type"] == t]
        answerable_typed = [r for r in typed if r["answerable"]]
        by_type[t] = {**_aggregate(answerable_typed), "total": len(typed)}

    overall = _aggregate(answerable_rows)
    n_rows = len(rows)

    def cost_mean(key: str) -> float:
        return sum(r["cost"][key] for r in rows) / n_rows

    summary = {
        "label": label,
        "corpus": corpus,
        "config": {
            "retrievers": list(dict.fromkeys(retrievers)),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "candidates": max(candidates, depth),
            "weights": weights,
            "dense_backend": (index_info.get("dense") or {}).get("backend"),
            "bm25_stopwords": bm25_module.stopwords_enabled() if "bm25" in retrievers else None,
            "generate_answers": generate_answers
        },
        "total_questions": n_rows,
        "answerable_questions": len(answerable_rows),
        "hit_rates": {f"hit_rate_at_{n}": overall[f"hit_rate_at_{n}"] for n in CUTOFFS},
        "recall": {f"recall_at_{n}": overall[f"recall_at_{n}"] for n in CUTOFFS},
        "hit_rate_at_k": _mean([float(r["hit_at_k"]) for r in answerable_rows]),
        "recall_at_k": _mean([r["recall_at_k"] for r in answerable_rows]),
        "by_type": by_type,
        "questions_with_unmatched_hops": [r["id"] for r in rows if r["unmatched_hops"]],
        "index": index_info,
        "cost_per_question": {
            "retrieval_ms": cost_mean("retrieval_s") * 1000,
            "latency_ms": cost_mean("latency_s") * 1000,
            "llm_calls": cost_mean("llm_calls"),
            "prompt_tokens": cost_mean("prompt_tokens"),
            "completion_tokens": cost_mean("completion_tokens")
        }
    }

    return {"summary": summary, "results": rows}


def regressions(
    base: dict[str, Any],
    new: dict[str, Any]
) -> list[dict[str, Any]]:
    """Questions where `new` ranks the first gold chunk worse than `base`
    (or loses it). This is the "adding a retriever made things worse" list."""
    base_rows = {r["id"]: r for r in base["results"] if r["answerable"]}
    worse = []

    for r in new["results"]:
        b = base_rows.get(r["id"])

        if not b:
            continue

        before, after = b["first_gold_rank"], r["first_gold_rank"]
        lost = before is not None and (after is None or after > before)

        if lost:
            worse.append({
                "id": r["id"],
                "type": r["type"],
                "question": r["question"],
                "rank_before": before,
                "rank_after": after,
                "recall_before": b[f"recall_at_{base['summary']['config']['top_k']}"],
                "recall_after": r[f"recall_at_{new['summary']['config']['top_k']}"]
            })

    return worse
