"""Diagnose MCQ Reranking pools: pool size, gold alignment, real random baseline.

For every MCQ Reranking task (GRILE / JuRo / WWTBM / RoMedQA) this measures:
  - distribution of |top_ranked| per query (min/median/max, empirical CDF)
  - fraction of queries whose gold corpus-id is missing from their pool
  - real random baseline (mean over queries of 1/k for single-gold, or
    mean(H_k / k) for MAP)

USAGE:
    ./apptainer-exec-romteb.sh scripts/mcq_reranking_diag.py
    ./apptainer-exec-romteb.sh scripts/mcq_reranking_diag.py --tasks WWTBMRoQAReranking
    ./apptainer-exec-romteb.sh scripts/mcq_reranking_diag.py --update_config
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


HF_PATHS = {
    "GrileGrammarReranking": "alina0195/romteb-grile-reranking",
    "JuRoLegalExamReranking": "alina0195/romteb-juro-legal-reranking",
    "WWTBMRoQAReranking": "alina0195/romteb-wwtbm-ro",
    "RoMedQAv2Reranking": "alina0195/romteb-romedqa-v2-reranking",
}


def _harmonic(n: int) -> float:
    return sum(1.0 / i for i in range(1, n + 1))


def _load_task_frames(hf_path: str):
    from romteb._hf_datasets import load_dataset

    frames: dict[str, dict] = {}
    # `default` is where MTEB puts qrels; the others are queries / corpus / pool.
    for cfg in ("queries", "corpus", "top_ranked", "default"):
        try:
            ds = load_dataset(hf_path, cfg, split="test")
        except Exception:
            try:
                dsd = load_dataset(hf_path, cfg)
                if hasattr(dsd, "keys"):
                    ds = dsd[list(dsd.keys())[0]]
                else:
                    ds = dsd
            except Exception:
                continue
        frames[cfg] = ds
    return frames


def _pool_sizes_and_alignment(hf_path: str) -> dict:
    from romteb._hf_datasets import load_dataset

    frames = _load_task_frames(hf_path)
    tr = frames.get("top_ranked")
    qrels = frames.get("default") or frames.get("qrels")
    if tr is None:
        return {"error": f"no top_ranked config for {hf_path}"}

    col_qid = "query-id" if "query-id" in tr.column_names else "query_id"
    col_cids = "corpus-ids" if "corpus-ids" in tr.column_names else "corpus_ids"

    top_ranked_by_q = {row[col_qid]: set(row[col_cids]) for row in tr}
    pool_sizes = [len(v) for v in top_ranked_by_q.values()]

    gold_by_q: dict = {}
    if qrels is not None:
        q_col = "query-id" if "query-id" in qrels.column_names else "query_id"
        c_col = "corpus-id" if "corpus-id" in qrels.column_names else "corpus_id"
        s_col = "score" if "score" in qrels.column_names else None
        for row in qrels:
            if s_col and int(row[s_col]) <= 0:
                continue
            gold_by_q.setdefault(row[q_col], set()).add(row[c_col])

    missing_gold = 0
    n_gold_per_q: list[int] = []
    if gold_by_q:
        for qid, golds in gold_by_q.items():
            pool = top_ranked_by_q.get(qid, set())
            n_gold_per_q.append(len(golds & pool))
            if not (golds & pool):
                missing_gold += 1

    n_q = len(pool_sizes)
    e_acc = sum(1.0 / max(k, 1) for k in pool_sizes) / max(n_q, 1)
    e_map_single = sum(_harmonic(k) / k for k in pool_sizes) / max(n_q, 1)

    dist = Counter(pool_sizes)
    return {
        "n_queries": n_q,
        "n_qrels": sum(len(v) for v in gold_by_q.values()) if gold_by_q else None,
        "pool_min": min(pool_sizes) if pool_sizes else None,
        "pool_median": int(median(pool_sizes)) if pool_sizes else None,
        "pool_max": max(pool_sizes) if pool_sizes else None,
        "pool_size_hist": sorted(dist.items()),
        "queries_with_gold_missing_from_pool": missing_gold,
        "mean_gold_in_pool_per_q": (
            sum(n_gold_per_q) / len(n_gold_per_q) if n_gold_per_q else None
        ),
        "E[acc@1]_random": e_acc,
        "E[MAP]_random_single_gold": e_map_single,
    }


def _format_report(stats: dict[str, dict]) -> str:
    lines = ["MCQ Reranking pool diagnostic", "=" * 60, ""]
    for task, s in stats.items():
        lines.append(f"[{task}]")
        if "error" in s:
            lines.append(f"  ERROR: {s['error']}")
            continue
        lines.append(f"  n_queries = {s['n_queries']}  n_qrels = {s['n_qrels']}")
        lines.append(
            f"  pool size: min={s['pool_min']}  median={s['pool_median']}  max={s['pool_max']}"
        )
        lines.append(f"  histogram (pool_size, count): {s['pool_size_hist'][:10]}")
        if s.get("n_qrels") is None:
            lines.append("  (no qrels file loaded; skipping gold-alignment check)")
        else:
            n = s["n_queries"]
            miss = s["queries_with_gold_missing_from_pool"]
            frac = miss / n if n else 0
            lines.append(
                f"  queries with gold OUTSIDE their pool: {miss} / {n} "
                f"({frac:.3%})   <-- if >0 the reranking task cannot achieve accuracy=1"
            )
            mean_g = s.get("mean_gold_in_pool_per_q")
            if mean_g is not None:
                lines.append(f"  mean gold-in-pool per query: {mean_g:.3f}")
        lines.append(f"  E[acc@1] random uniform: {s['E[acc@1]_random']:.4f}")
        lines.append(f"  E[MAP]  random uniform (single-gold): {s['E[MAP]_random_single_gold']:.4f}")
        lines.append("")
    return "\n".join(lines)


def _patch_baseline(task_name: str, k_median: int, acc: float, map_val: float) -> None:
    path = ROOT / "romteb" / "eval_config.py"
    text = path.read_text(encoding="utf-8")
    pattern = re.escape(f'"{task_name}":') + r' \{"k": \d+, "accuracy": [0-9.]+, "map_at_1000": [0-9.]+\}'
    repl = (
        f'"{task_name}": {{"k": {k_median}, "accuracy": {acc:.3f}, '
        f'"map_at_1000": {map_val:.3f}}}'
    )
    new, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        print(f"[warn] could not patch {task_name} baseline in {path}")
        return
    path.write_text(new, encoding="utf-8")
    print(f"Updated {path.relative_to(ROOT)}: {task_name} k={k_median} acc={acc:.3f} MAP={map_val:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(HF_PATHS.keys()),
        help="Subset of MCQ reranking tasks (default: all four)",
    )
    parser.add_argument("--out", default="results/mcq_reranking_diag.txt")
    parser.add_argument(
        "--json_out",
        default="results/mcq_reranking_diag.json",
        help="Machine-readable JSON dump of the stats.",
    )
    parser.add_argument(
        "--update_config",
        action="store_true",
        help=(
            "Rewrite RERANKING_RANDOM_BASELINE from the measured per-query k. "
            "RoMedQA is skipped (multi-relevant)."
        ),
    )
    args = parser.parse_args()

    stats: dict[str, dict] = {}
    for task in args.tasks:
        if task not in HF_PATHS:
            print(f"[skip] unknown task {task!r}")
            continue
        print(f"[diag] {task} <- {HF_PATHS[task]}")
        stats[task] = _pool_sizes_and_alignment(HF_PATHS[task])

    report = _format_report(stats)
    print(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")

    json_out = Path(args.json_out)
    json_out.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {json_out}")

    if args.update_config:
        for task, s in stats.items():
            if "error" in s or s.get("pool_median") is None:
                continue
            if task == "RoMedQAv2Reranking":
                print(f"[skip] {task}: multi-relevant, MAP baseline is not 1/k")
                continue
            _patch_baseline(
                task,
                int(s["pool_median"]),
                float(s["E[acc@1]_random"]),
                float(s["E[MAP]_random_single_gold"]),
            )


if __name__ == "__main__":
    main()
