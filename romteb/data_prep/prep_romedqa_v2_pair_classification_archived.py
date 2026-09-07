"""ARCHIVED — RoMedQA PairClassification. Superseded by retrieval + reranking (prep_romedqa_v2.py).

Prepare RoMedQA_v2 MCQ into MTEB PairClassification format for RoMTEB.

By default reads the already-uploaded BEIR reranking repo
``alina0195/romteb-romedqa-v2-reranking`` and converts it to one HF row per
(question, option) pair. Pass ``--from_source`` to rebuild from
``craciuncg/RoMedQA_v2`` instead.
"""

from __future__ import annotations

import argparse
import re

from romteb._hf_datasets import DatasetDict, load_dataset

from romteb.data_prep._beir_to_pair_classification import (
    beir_reranking_to_pairs,
    load_beir_reranking,
    pack_pair_classification,
    print_pair_stats,
)
from romteb.data_prep._common import push_to_hub


SOURCE_REPO = "craciuncg/RoMedQA_v2"
SOURCE_RERANKING_REPO = "alina0195/romteb-romedqa-v2-reranking"
TARGET_REPO = "alina0195/romteb-romedqa-v2-pair-classification"


def _split_question_and_options(question_text: str) -> tuple[str, dict[str, str]]:
    lines = [ln.strip() for ln in (question_text or "").splitlines() if ln.strip()]
    stem_lines: list[str] = []
    options: dict[str, str] = {}
    found_options = False

    for ln in lines:
        m = re.match(r"^([1-5])\.\s*(.+)$", ln)
        if m:
            found_options = True
            options[m.group(1)] = m.group(2).strip()
        elif not found_options:
            stem_lines.append(ln)

    stem = " ".join(stem_lines).strip()
    if not stem:
        stem = (question_text or "").strip()
    return stem, options


def _extract_correct_choices(answer_value) -> list[str]:
    return sorted(set(re.findall(r"[1-5]", str(answer_value))))


def _build_from_source(ds) -> tuple[list[str], list[str], list[int], dict[str, int]]:
    sentence1: list[str] = []
    sentence2: list[str] = []
    labels: list[int] = []
    skipped = 0
    kept = 0

    for row in ds:
        stem, options = _split_question_and_options((row.get("question") or "").strip())
        if not stem or len(options) < 2:
            skipped += 1
            continue

        correct = {c for c in _extract_correct_choices(row.get("answer")) if c in options}
        if not correct:
            skipped += 1
            continue

        for choice in sorted(options.keys()):
            sentence1.append(stem)
            sentence2.append(options[choice])
            labels.append(1 if choice in correct else 0)
        kept += 1

    stats = {
        "kept": kept,
        "skipped": skipped,
        "pairs": len(labels),
        "positives": sum(labels),
    }
    return sentence1, sentence2, labels, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_source",
        action="store_true",
        help=f"Rebuild from {SOURCE_REPO} instead of {SOURCE_RERANKING_REPO}",
    )
    parser.add_argument(
        "--from_hf",
        default=None,
        help="BEIR reranking HF repo to convert (default: romedqa-v2 reranking repo)",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.from_source:
        print(f"Loading {SOURCE_REPO} (split=test)")
        ds = load_dataset(SOURCE_REPO, split="test")
        sentence1, sentence2, labels, stats = _build_from_source(ds)
    else:
        repo = args.from_hf or SOURCE_RERANKING_REPO
        print(f"Loading BEIR reranking repo: {repo}")
        corpus, queries, qrels, top_ranked = load_beir_reranking(repo)
        sentence1, sentence2, labels, stats = beir_reranking_to_pairs(
            corpus, queries, qrels, top_ranked
        )

    print_pair_stats("RoMedQA_v2 PairClassification", stats, sentence1, sentence2, labels)

    if args.dry_run:
        return

    out = DatasetDict({"test": pack_pair_classification(sentence1, sentence2, labels)})
    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message="RoMTEB RoMedQA_v2 MCQ (PairClassification, one row per pair)",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
