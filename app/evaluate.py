"""Command-line evaluation. Same code as the UI/API (app/evaluation.py).

Examples (from the repository root):

  # one configuration
  python -m app.evaluate run --corpus kestrel --retrievers dense,bm25 --top-k 10

  # every configuration of the V3 report, both corpora, one command
  python -m app.evaluate matrix

  # same, with generated answers (uses LLM quota) and a weighted RRF variant
  python -m app.evaluate run --corpus kestrel --retrievers dense,bm25,graph --weights dense=1,bm25=1,graph=2

Dense runs in strict mode by default: if sentence-transformers cannot load, the run
stops instead of silently measuring TF-IDF. Use --allow-tfidf-fallback to override
(the result file then says which backend was really used).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import bm25 as bm25_module
from . import core
from .evaluation import CUTOFFS, TYPE_ORDER, evaluate_corpus, regressions
from .retrieval import graph_available

# V2's final chunking per corpus (REPORT V2 section 7): V3 is measured against V2
# on the same footing.
V2_CHUNKING = {"kestrel": (800, 10), "own": (500, 20)}
DEFAULT_OUT = core.DATA / "results" / "v3"


def parse_retrievers(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_weights(text: str | None) -> dict[str, float] | None:
    if not text:
        return None

    out = {}

    for part in text.split(","):
        name, _, value = part.partition("=")
        out[name.strip()] = float(value)

    return out


def run_name(retrievers: list[str], stopwords: bool = True, weights=None) -> str:
    name = "+".join(retrievers)

    if "bm25" in retrievers and not stopwords:
        name += " (no-stopwords)"

    if weights:
        name += " (weighted " + ",".join(f"{k}={v:g}" for k, v in weights.items()) + ")"

    return name


def save(result: dict[str, Any], out_dir: Path, corpus: str, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9+=,\-]+", "_", name).strip("_")
    path = out_dir / f"{corpus}__{safe}.json"
    core.save_json(path, result)

    return path


def execute(
    corpus: str, retrievers: list[str], args, stopwords: bool = True, weights=None,
    size: int | None = None, overlap: int | None = None
) -> tuple[str, dict[str, Any], Path]:
    size = size or (args.chunk_size or V2_CHUNKING[corpus][0])
    overlap = overlap if overlap is not None else (
        args.overlap if args.overlap is not None else V2_CHUNKING[corpus][1]
    )
    bm25_module.set_stopwords(stopwords)
    name = run_name(retrievers, stopwords, weights)
    started = time.perf_counter()

    def progress(i: int, n: int, qid: str):
        if sys.stdout.isatty():
            print(f"  [{corpus} | {name}] {i}/{n} {qid}", end="\r", flush=True)

    questions = None

    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in core.load_questions(corpus) if q.get("id") in wanted]

    result = evaluate_corpus(
        corpus, size, overlap, args.top_k, retrievers, questions=questions,
        generate_answers=args.generate,
        rrf_k=args.rrf_k, candidates=args.candidates, weights=weights,
        strict_dense=not args.allow_tfidf_fallback,
        label=name, progress=progress
    )
    result["summary"]["wall_seconds"] = time.perf_counter() - started
    path = save(result, Path(args.out_dir), corpus, name)
    bm25_module.set_stopwords(True)
    print(f"  [{corpus} | {name}] done -> {path.relative_to(core.ROOT) if path.is_relative_to(core.ROOT) else path}      ")

    return name, result, path


def fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def results_table(runs: list[tuple[str, str, dict[str, Any]]]) -> str:
    """configurations x corpora x question types, Hit@k and Recall@k."""
    head = "| Corpus | Configuration | Type | n | " + " | ".join(f"Hit@{n}" for n in CUTOFFS) + " | " + " | ".join(f"Recall@{n}" for n in CUTOFFS) + " |\n"
    head += "|---|---|---|---:|" + "---:|" * (2 * len(CUTOFFS)) + "\n"
    lines = []

    for corpus, name, result in runs:
        s = result["summary"]
        lines.append(
            f"| {corpus} | {name} | **ALL** | {s['answerable_questions']} | "
            + " | ".join(fmt(s["hit_rates"][f"hit_rate_at_{n}"]) for n in CUTOFFS) + " | "
            + " | ".join(fmt(s["recall"][f"recall_at_{n}"]) for n in CUTOFFS) + " |"
        )

        for t, v in s["by_type"].items():
            if v["count"] == 0:
                continue

            lines.append(
                f"| {corpus} | {name} | {t} | {v['count']} | "
                + " | ".join(fmt(v[f"hit_rate_at_{n}"]) for n in CUTOFFS) + " | "
                + " | ".join(fmt(v[f"recall_at_{n}"]) for n in CUTOFFS) + " |"
            )

    return head + "\n".join(lines) + "\n"


def cost_table(runs) -> str:
    out = "| Corpus | Configuration | Retrieval ms/question | Total ms/question | LLM calls/question | Prompt tok/question | Completion tok/question | Dense backend | Index build |\n"
    out += "|---|---|---:|---:|---:|---:|---:|---|---|\n"

    for corpus, name, result in runs:
        s = result["summary"]
        c = s["cost_per_question"]
        idx = s["index"]
        build = "; ".join(
            f"{k}: {(v.get('build_seconds') or 0):.2f}s"
            + (" (cached)" if v.get("embedding_cache_hit") or v.get("source") in ("disk", "memory") else "")
            for k, v in idx.items()
        ) or "–"
        out += (
            f"| {corpus} | {name} | {c['retrieval_ms']:.1f} | {c['latency_ms']:.1f} | {c['llm_calls']:.2f} | "
            f"{c['prompt_tokens']:.0f} | {c['completion_tokens']:.0f} | {s['config']['dense_backend'] or '–'} | {build} |\n"
        )

    return out


def regression_report(runs) -> str:
    by = {(corpus, name): result for corpus, name, result in runs}
    pairs = [
        ("dense", "dense+bm25"),
        ("dense+bm25", "dense+bm25+graph"),
        ("dense", "dense+bm25+graph"),
    ]
    out = ""

    for corpus in sorted({c for c, _ in by}):
        for a, b in pairs:
            if (corpus, a) in by and (corpus, b) in by:
                worse = regressions(by[(corpus, a)], by[(corpus, b)])
                out += f"\n**{corpus}: `{a}` → `{b}`**: {len(worse)} question(s) where the first gold chunk ranks worse.\n"

                for w in worse:
                    out += (
                        f"- `{w['id']}` ({w['type']}): rank {w['rank_before']} → {w['rank_after']}"
                        f" (recall@k {fmt(w['recall_before'])} → {fmt(w['recall_after'])}) — {w['question']}\n"
                    )

    return out or "\n(no comparable pairs were run)\n"


def write_summary(runs, out_dir: Path) -> Path:
    unmatched = {
        (corpus, name): r["summary"]["questions_with_unmatched_hops"]
        for corpus, name, r in runs
    }
    md = "# V3 evaluation summary\n\n"
    md += f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Chunking per corpus: " + ", ".join(
        f"{c} {s}/{o}" for c, (s, o) in V2_CHUNKING.items()
    ) + ". Hit@k = at least one hop retrieved; Recall@k = share of hops retrieved. not_in_corpus questions are excluded from both._\n\n"
    fallback = sorted({
        name for _, name, r in runs
        if r["summary"]["config"]["dense_backend"] not in (None, "sentence-transformers")
    })

    if fallback:
        md += (
            "> **WARNING: the dense retriever fell back to TF-IDF for: " + ", ".join(fallback)
            + ". These numbers are NOT comparable with V2. Install sentence-transformers and rerun without --allow-tfidf-fallback.**\n\n"
        )

    md += "## Results: configurations × corpora × question types\n\n" + results_table(runs)
    md += "\n## Cost per question and index build\n\n" + cost_table(runs)
    md += "\n## Where adding a retriever made things worse\n" + regression_report(runs)
    flagged = {k: v for k, v in unmatched.items() if v}

    if flagged:
        md += "\n## Questions whose gold quote matches no chunk (check chunking/annotation)\n\n"

        for (corpus, name), ids in flagged.items():
            md += f"- {corpus} / {name}: {', '.join(map(str, ids))}\n"

    path = out_dir / "summary.md"
    path.write_text(md, encoding="utf-8")
    core.save_json(out_dir / "summary.json", [
        {"corpus": c, "configuration": n, "summary": r["summary"]} for c, n, r in runs
    ])

    return path


def cmd_run(args) -> int:
    retrievers = parse_retrievers(args.retrievers)
    name, result, _ = execute(
        args.corpus, retrievers, args, stopwords=not args.no_stopwords,
        weights=parse_weights(args.weights)
    )
    out_dir = Path(args.out_dir)
    write_summary([(args.corpus, name, result)], out_dir)
    print(results_table([(args.corpus, name, result)]))

    return 0


def cmd_matrix(args) -> int:
    graph = graph_available()
    plan: list[tuple[str, list[str], bool]] = []

    for corpus in ("kestrel", "own"):
        plan += [(corpus, ["dense"], True), (corpus, ["dense", "bm25"], True)]

        if corpus == "kestrel":
            if graph:
                plan.append((corpus, ["dense", "bm25", "graph"], True))
            else:
                print("NOTE: app/graph_retriever.py not found: the dense+bm25+graph run is skipped.")

        if not args.required_only:
            plan.append((corpus, ["bm25"], True))
            plan.append((corpus, ["bm25"], False))  # stop-word ablation

            if corpus == "kestrel" and graph:
                plan.append((corpus, ["graph"], True))

    runs = []

    for corpus, retrievers, stop in plan:
        name, result, _ = execute(corpus, retrievers, args, stopwords=stop)
        runs.append((corpus, name, result))

    path = write_summary(runs, Path(args.out_dir))
    print(f"\nSummary written to {path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG V3 retrieval evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--top-k", type=int, default=10)
        p.add_argument("--chunk-size", type=int, default=None, help="default: V2 final value per corpus")
        p.add_argument("--overlap", type=int, default=None, help="default: V2 final value per corpus")
        p.add_argument("--rrf-k", type=int, default=60)
        p.add_argument("--candidates", type=int, default=10, help="how many results each retriever gives to the fusion (D-18)")
        p.add_argument("--ids", default=None, help="only these question ids, e.g. q01,q02,q03 (V2 sanity check: q01..q10)")
        p.add_argument("--generate", action="store_true", help="also generate answers (LLM calls)")
        p.add_argument("--allow-tfidf-fallback", action="store_true")
        p.add_argument("--out-dir", default=str(DEFAULT_OUT))

    run = sub.add_parser("run", help="evaluate one configuration")
    run.add_argument("--corpus", choices=["kestrel", "own"], required=True)
    run.add_argument("--retrievers", default="dense,bm25", help="comma list among dense,bm25,graph")
    run.add_argument("--weights", default=None, help="weighted RRF, e.g. dense=1,bm25=1,graph=2")
    run.add_argument("--no-stopwords", action="store_true", help="BM25 without the stop-word list")
    common(run)
    run.set_defaults(func=cmd_run)

    matrix = sub.add_parser("matrix", help="all configurations of the V3 report, both corpora")
    matrix.add_argument("--required-only", action="store_true", help="only the 3 required configurations")
    common(matrix)
    matrix.set_defaults(func=cmd_matrix)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        return args.func(args)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
