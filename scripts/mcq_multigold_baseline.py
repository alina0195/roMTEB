"""Simulate random-uniform MAP@1000 for multi-gold MCQ reranking (RoMedQA, JuRo).

For each query with `g` gold options in a pool of size `k`, we permute the
pool uniformly at random `n_trials` times and compute MAP for the top-`k`
retrieval. Mean over queries and trials gives the empirical E[MAP] baseline
that a uniform random ranker would produce.

USAGE:
    ./apptainer-exec-romteb.sh scripts/mcq_multigold_baseline.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from romteb._hf_datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]

HF_PATHS = {
    "GrileGrammarReranking": "alina0195/romteb-grile-reranking",
    "JuRoLegalExamReranking": "alina0195/romteb-juro-legal-reranking",
    "WWTBMRoQAReranking": "alina0195/romteb-wwtbm-ro",
    "RoMedQAv2Reranking": "alina0195/romteb-romedqa-v2-reranking",
}


def _load(hf_path: str) -> tuple[dict[str, set], dict[str, set]]:
    tr = load_dataset(hf_path, "top_ranked", split="test")
    qrels = load_dataset(hf_path, "default", split="test")
    q_col = "query-id" if "query-id" in tr.column_names else "query_id"
    c_col = "corpus-ids" if "corpus-ids" in tr.column_names else "corpus_ids"
    pool = {row[q_col]: set(row[c_col]) for row in tr}

    q_col2 = "query-id" if "query-id" in qrels.column_names else "query_id"
    c_col2 = "corpus-id" if "corpus-id" in qrels.column_names else "corpus_id"
    s_col = "score" if "score" in qrels.column_names else None
    gold: dict[str, set] = {}
    for row in qrels:
        if s_col and int(row[s_col]) <= 0:
            continue
        gold.setdefault(row[q_col2], set()).add(row[c_col2])
    return pool, gold


def _map_at_k(order: list[str], gold: set[str]) -> float:
    hits = 0
    ap = 0.0
    for i, cid in enumerate(order, 1):
        if cid in gold:
            hits += 1
            ap += hits / i
    if not gold:
        return 0.0
    return ap / len(gold)


def _acc_at_1(order: list[str], gold: set[str]) -> int:
    return 1 if order and order[0] in gold else 0


def simulate(pool: dict, gold: dict, n_trials: int, seed: int) -> dict:
    rng = random.Random(seed)
    maps: list[float] = []
    accs: list[int] = []
    for qid, cand in pool.items():
        g = gold.get(qid, set())
        if not g:
            continue
        cand_list = list(cand)
        for _ in range(n_trials):
            rng.shuffle(cand_list)
            maps.append(_map_at_k(cand_list, g))
            accs.append(_acc_at_1(cand_list, g))
    n = len(maps)
    return {
        "n_samples": n,
        "mean_MAP": sum(maps) / max(n, 1),
        "mean_acc_at_1": sum(accs) / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--out", default="results/mcq_random_baseline.json")
    args = parser.parse_args()

    out: dict[str, dict] = {}
    for task, path in HF_PATHS.items():
        pool, gold = _load(path)
        print(f"[sim] {task}: n_queries_with_gold={len(gold)}")
        stats = simulate(pool, gold, args.n_trials, args.seed)
        print(
            f"  E[MAP]={stats['mean_MAP']:.4f}  "
            f"E[acc@1]={stats['mean_acc_at_1']:.4f}  "
            f"n_samples={stats['n_samples']}"
        )
        out[task] = stats

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
