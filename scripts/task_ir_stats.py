"""Corpus / query sizes for RoMTEB retrieval and reranking tasks.

No GPU. Writes ``romteb_ir_task_stats.csv`` (default: repo root or --out).

USAGE:
    ./apptainer-exec-romteb.sh scripts/task_ir_stats.py
    ./apptainer-exec-romteb.sh scripts/task_ir_stats.py --out results/romteb_ir_task_stats.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from romteb.benchmark import ROMTEB_TASKS
from romteb.eval_config import metric_of
from romteb.tasks.retrieval.msmarco_ro import MSMarcoRoRetrieval

IR_TYPES = {"Retrieval", "Reranking"}


def _split_data(task, split: str):
    dataset = getattr(task, "dataset", None)
    if not dataset:
        return None
    if split in dataset:
        blob = dataset[split]
        if isinstance(blob, dict) and "queries" in blob:
            return blob
    # multilingual / subset layout: subset -> split -> {queries, corpus}
    for _subset, splits in dataset.items():
        if not isinstance(splits, dict):
            continue
        blob = splits.get(split) if isinstance(splits, dict) else None
        if isinstance(blob, dict) and "queries" in blob:
            return blob
        if hasattr(splits, "keys") and "queries" in splits:
            return splits
    return None


def _len(obj) -> int:
    if obj is None:
        return 0
    if hasattr(obj, "__len__"):
        try:
            return len(obj)
        except TypeError:
            pass
    if isinstance(obj, dict):
        return len(obj)
    return 0


def _stats_for_task(task) -> dict | None:
    name = task.metadata.name
    task_type = task.metadata.type
    if task_type not in IR_TYPES:
        return None
    split = (task.metadata.eval_splits or ["test"])[0]
    print(f"[ir-stats] loading {name} ...")
    try:
        task.load_data()
    except Exception as exc:
        print(f"[ir-stats] skip {name}: {exc}")
        return None
    blob = _split_data(task, split)
    n_queries = n_corpus = n_qrels = n_pool = None
    if blob:
        n_queries = _len(blob.get("queries"))
        n_corpus = _len(blob.get("corpus"))
        rel = blob.get("relevant_docs") or {}
        n_qrels = _len(rel)
        top = blob.get("top_ranked")
        n_pool = _len(top) if top is not None else None
    return {
        "task": name,
        "task_type": task_type,
        "metric": metric_of(task_type, name),
        "split": split,
        "n_queries": n_queries,
        "n_corpus": n_corpus,
        "n_qrels": n_qrels,
        "n_top_ranked": n_pool,
        "held_out": name == "MSMarcoRoRetrieval",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=None,
        help="CSV path (default: <repo>/romteb_ir_task_stats.csv)",
    )
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "romteb_ir_task_stats.csv"

    tasks = [t for t in ROMTEB_TASKS if getattr(t.metadata, "type", "") in IR_TYPES]
    try:
        held = MSMarcoRoRetrieval()
        tasks.append(held)
    except Exception as exc:
        print(f"[ir-stats] MSMarcoRoRetrieval not instantiated: {exc}")

    rows = []
    for task in tasks:
        row = _stats_for_task(task)
        if row:
            rows.append(row)
            print(
                f"  {row['task']}: n_queries={row['n_queries']} "
                f"n_corpus={row['n_corpus']}"
            )

    if not rows:
        raise SystemExit("no IR stats collected")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
