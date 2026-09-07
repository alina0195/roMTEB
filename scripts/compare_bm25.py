"""Compare BM25 baseline against dense encoders on Retrieval + Reranking.

Purpose: implement the sanity check from §1.3 / §1.6 of the review
(evaluation_protocol.md). Any dense model that lands **below** BM25 on any
retrieval task is a strong signal for a prompting bug — the model's raw
embeddings are fine (classification/reranking are usually normal) but the
query↔passage space isn't aligned because the required prefix/instruction
was skipped.

Usage:
  ./apptainer-exec-romteb.sh python3 scripts/compare_bm25.py \\
      --results results \\
      [--only Retrieval|Reranking|both]  # default both
      [--tsv summary.tsv]

BM25 baseline lives at `results/romteb__bm25s-ro/<version>/*.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BM25_DIR_NAME = "romteb__bm25s-ro"

RETRIEVAL_TASKS = {
    "WebFAQRetrieval",
    "WikipediaRetrievalMultilingual",
    "XQuADRetrieval",
    "RoDTALLawsRetrieval",
    "MQARoCQARetrieval",
}

RERANKING_TASKS = {
    "JuRoLegalExamReranking",
    "WWTBMRoQAReranking",
    "RoMedQAv2Reranking",
    "GrileGrammarReranking",
}


def _main_score(json_path: Path) -> float | None:
    """Return the primary metric from an MTEB result file, or None."""
    with json_path.open() as fp:
        d = json.load(fp)
    scores = d.get("scores", {})
    if not isinstance(scores, dict):
        return None
    for _, entries in scores.items():
        if not entries:
            continue
        e = entries[0] if isinstance(entries, list) else entries
        ms = e.get("main_score")
        if ms is not None:
            return float(ms)
    return None


def _collect(results_dir: Path, task_names: set[str]) -> dict[str, dict[str, float]]:
    """Return {model_dir_name: {task_name: main_score}} for the given tasks."""
    out: dict[str, dict[str, float]] = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for ver_dir in model_dir.iterdir():
            if not ver_dir.is_dir():
                continue
            for f in ver_dir.glob("*.json"):
                name = f.stem
                if name not in task_names:
                    continue
                s = _main_score(f)
                if s is None:
                    continue
                out.setdefault(model_dir.name, {})[name] = s
    return out


def _print_table(
    scores: dict[str, dict[str, float]],
    task_order: list[str],
    bm25_row: dict[str, float],
    tsv_out: Path | None,
    severe_gap: float,
) -> None:
    severe: dict[str, list[tuple[str, float, float]]] = {}
    marginal: dict[str, list[tuple[str, float, float]]] = {}
    header = f"{'Model':55s}  " + "  ".join(f"{t[:14]:>14s}" for t in task_order)
    print(header)
    bm_cells = "  ".join(f"{bm25_row.get(t, float('nan')):>14.3f}" for t in task_order)
    print(f"{'BM25 (baseline)':55s}  {bm_cells}")
    print("-" * len(header))
    tsv_lines: list[str] = ["model\t" + "\t".join(task_order)]
    tsv_lines.append(
        "BM25\t" + "\t".join(f"{bm25_row.get(t, ''):.3f}" if t in bm25_row else "" for t in task_order)
    )
    for m in sorted(scores):
        if m == BM25_DIR_NAME:
            continue
        row = scores[m]
        cells = []
        tsv_row = [m]
        for t in task_order:
            v = row.get(t)
            b = bm25_row.get(t)
            if v is None:
                cells.append(f"{'--':>14s}")
                tsv_row.append("")
                continue
            tsv_row.append(f"{v:.3f}")
            if b is None or v >= b:
                cells.append(f"{v:>14.3f}")
                continue
            gap = b - v
            if gap >= severe_gap:
                cells.append(f"{v:>13.3f}!")
                severe.setdefault(m, []).append((t, v, b))
            else:
                cells.append(f"{v:>13.3f}*")
                marginal.setdefault(m, []).append((t, v, b))
        print(f"{m:55s}  {'  '.join(cells)}")
        tsv_lines.append("\t".join(tsv_row))
    print(
        f"\n! = severely below BM25 (gap >= {severe_gap:.2f}, likely prompt bug)"
        f"  |  * = marginally below (noise near saturated BM25)\n"
    )
    if severe:
        print("Severely below BM25 (fix prompt / loader / instruction template):")
        for m, tasks in sorted(severe.items(), key=lambda x: -len(x[1])):
            entries = ", ".join(f"{t}({v:.3f} vs {b:.3f})" for t, v, b in tasks)
            print(f"  - {m}  [{len(tasks)} task(s)]: {entries}")
    else:
        print("No dense models severely below BM25.")
    if marginal:
        print("\nMarginally below BM25 (task likely trivially lexical, e.g., XQuAD):")
        for m, tasks in sorted(marginal.items(), key=lambda x: -len(x[1])):
            entries = ", ".join(f"{t}({v:.3f} vs {b:.3f})" for t, v, b in tasks)
            print(f"  - {m}  [{len(tasks)} task(s)]: {entries}")
    if tsv_out is not None:
        tsv_out.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
        print(f"\nWrote TSV: {tsv_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument(
        "--only",
        choices=("Retrieval", "Reranking", "both"),
        default="both",
        help="Which task category to include (default: both)",
    )
    ap.add_argument("--tsv", type=Path, default=None, help="Optional TSV output")
    ap.add_argument(
        "--severe-gap",
        type=float,
        default=0.05,
        help="Absolute gap below BM25 that flips a model from marginal (*) to severe (!). Default 0.05.",
    )
    args = ap.parse_args()

    if args.only == "Retrieval":
        task_names = RETRIEVAL_TASKS
    elif args.only == "Reranking":
        task_names = RERANKING_TASKS
    else:
        task_names = RETRIEVAL_TASKS | RERANKING_TASKS

    scores = _collect(args.results, task_names)
    bm25_row = scores.get(BM25_DIR_NAME)
    if not bm25_row:
        print(
            f"ERROR: no BM25 results found under {args.results / BM25_DIR_NAME}. "
            "Run `run_benchmark.py --model romteb/bm25s-ro --allow_cpu` first.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.only == "Retrieval":
        task_order = [t for t in sorted(RETRIEVAL_TASKS) if t in bm25_row]
    elif args.only == "Reranking":
        task_order = [t for t in sorted(RERANKING_TASKS) if t in bm25_row]
    else:
        task_order = sorted(t for t in task_names if t in bm25_row)

    _print_table(scores, task_order, bm25_row, args.tsv, args.severe_gap)


if __name__ == "__main__":
    main()
