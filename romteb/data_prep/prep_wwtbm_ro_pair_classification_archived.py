"""ARCHIVED — WWTBM PairClassification. Superseded by retrieval + reranking (prep_wwtbm_ro.py).

Prepare WWTBM-Romanian MCQ into MTEB PairClassification format for RoMTEB.

By default reads the already-uploaded BEIR reranking repo
``alina0195/romteb-wwtbm-ro`` and converts it to one HF row per
(question, option) pair. Pass ``--from_source`` to rebuild from
``WWTBM/wwtbm`` instead.
"""

from __future__ import annotations

import argparse
from collections import Counter

from romteb._hf_datasets import DatasetDict, load_dataset

from romteb.data_prep._beir_to_pair_classification import (
    beir_reranking_to_pairs,
    load_beir_reranking,
    pack_pair_classification,
    print_pair_stats,
)
from romteb.data_prep._common import push_to_hub


SOURCE_REPO = "WWTBM/wwtbm"
SOURCE_CONFIG = "Romanian"
SOURCE_SPLIT = "train"
SOURCE_RERANKING_REPO = "alina0195/romteb-wwtbm-ro"
TARGET_REPO = "alina0195/romteb-wwtbm-ro-pair-classification"


def _build_from_source(ds) -> tuple[list[str], list[str], list[int], dict[str, int | dict]]:
    sentence1: list[str] = []
    sentence2: list[str] = []
    labels: list[int] = []
    skipped_no_match = 0
    skipped_wrong_arity = 0
    culture_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()
    kept = 0

    for item in ds:
        question = (item.get("question") or "").strip()
        answers = list(item.get("answers") or [])
        gold = (item.get("correct_answer") or "").strip()

        if len(answers) != 4:
            skipped_wrong_arity += 1
            continue
        answers_norm = [(a or "").strip() for a in answers]
        if gold not in answers_norm:
            skipped_no_match += 1
            continue

        gold_pos = answers_norm.index(gold)
        for j, ans in enumerate(answers_norm):
            sentence1.append(question)
            sentence2.append(ans)
            labels.append(1 if j == gold_pos else 0)

        kept += 1
        ctx = item.get("cultural_context")
        if ctx:
            culture_counts[ctx] += 1
        diff = item.get("difficulty")
        if diff:
            difficulty_counts[diff] += 1

    stats: dict[str, int | dict] = {
        "kept": kept,
        "skipped": skipped_no_match + skipped_wrong_arity,
        "pairs": len(labels),
        "positives": sum(labels),
        "skipped_no_match": skipped_no_match,
        "skipped_wrong_arity": skipped_wrong_arity,
        "culture": dict(culture_counts),
        "difficulty": dict(difficulty_counts),
    }
    return sentence1, sentence2, labels, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_source",
        action="store_true",
        help=f"Rebuild from {SOURCE_REPO}:{SOURCE_CONFIG} instead of {SOURCE_RERANKING_REPO}",
    )
    parser.add_argument(
        "--from_hf",
        default=None,
        help="BEIR reranking HF repo to convert (default: romteb-wwtbm-ro)",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.from_source:
        print(f"Loading {SOURCE_REPO}:{SOURCE_CONFIG} ({SOURCE_SPLIT}) ...")
        ds = load_dataset(SOURCE_REPO, SOURCE_CONFIG, split=SOURCE_SPLIT)
        print(f"Loaded {len(ds):,} rows (columns: {ds.column_names})")
        sentence1, sentence2, labels, stats = _build_from_source(ds)
        if stats.get("culture"):
            print(f"  cultural_context: {stats['culture']}")
        if stats.get("difficulty"):
            print(f"  difficulty:       {stats['difficulty']}")
    else:
        repo = args.from_hf or SOURCE_RERANKING_REPO
        print(f"Loading BEIR reranking repo: {repo}")
        corpus, queries, qrels, top_ranked = load_beir_reranking(repo)
        sentence1, sentence2, labels, stats = beir_reranking_to_pairs(
            corpus, queries, qrels, top_ranked
        )

    print_pair_stats("WWTBM-Ro PairClassification", stats, sentence1, sentence2, labels)

    if args.dry_run:
        return

    out = DatasetDict({"test": pack_pair_classification(sentence1, sentence2, labels)})
    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message="RoMTEB WWTBM-Ro MCQ (PairClassification, one row per pair)",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
