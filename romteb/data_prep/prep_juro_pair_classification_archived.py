"""ARCHIVED — JuRo PairClassification. Superseded by retrieval + reranking (prep_juro.py).

Prepare JuRo legal MCQ into MTEB PairClassification format for RoMTEB.

Expected JuRo tree:
  JuRo/
    train_set/
    validation_set/
    test_set/

The benchmark is built from `test_set` only. For each question we emit one
(question, option) pair per answer choice, with label=1 for correct options and
label=0 for distractors. Each pair is stored as its own HF row with scalar
``sentence1``, ``sentence2``, and ``label`` columns.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from romteb._hf_datasets import DatasetDict

from romteb.data_prep._beir_to_pair_classification import (
    pack_pair_classification,
    print_pair_stats,
)
from romteb.data_prep._common import push_to_hub


TARGET_REPO = "alina0195/romteb-juro-legal-pair-classification"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _resolve_source_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"JuRo source dir not found: {p}")

    candidates = [
        WORKSPACE_ROOT / "datasets" / "JuRo",
        WORKSPACE_ROOT / "tmp_graf_repo" / "JuRo",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "JuRo source directory not found. Pass --source_dir or place JuRo under "
        f"{candidates[0]}."
    )


def _extract_options(options_blob: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in (options_blob or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-E])[\.\)]\s*(.+)$", line)
        if not m:
            continue
        letter, text = m.group(1), m.group(2).strip()
        if text:
            out.append((letter, text))
    seen = set()
    deduped: list[tuple[str, str]] = []
    for letter, text in out:
        if letter in seen:
            continue
        seen.add(letter)
        deduped.append((letter, text))
    return deduped


def _extract_gold_letters(answer_blob: str) -> list[str]:
    return sorted(set(re.findall(r"[A-E]", (answer_blob or "").upper())))


def _iter_test_rows(source_dir: Path):
    test_root = source_dir / "test_set"
    if not test_root.exists():
        raise FileNotFoundError(f"Missing test_set under {source_dir}")
    for csv_path in sorted(test_root.rglob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                yield csv_path, row[0].strip(), row[1].strip(), row[2], row[3]


def _build_pair_classification(source_dir: Path):
    sentence1: list[str] = []
    sentence2: list[str] = []
    labels: list[int] = []

    skipped = 0
    kept = 0
    for _path, subject, question, options_blob, answer_blob in _iter_test_rows(source_dir):
        options = _extract_options(options_blob)
        if len(options) < 2:
            skipped += 1
            continue

        gold_letters = set(_extract_gold_letters(answer_blob))
        if not gold_letters:
            skipped += 1
            continue

        option_map = {letter: text for letter, text in options}
        if not gold_letters & option_map.keys():
            skipped += 1
            continue

        query_text = f"[{subject}] {question}".strip()
        for letter, text in options:
            sentence1.append(query_text)
            sentence2.append(text)
            labels.append(1 if letter in gold_letters else 0)
        kept += 1

    stats = {
        "kept": kept,
        "skipped": skipped,
        "pairs": len(labels),
        "positives": sum(labels),
    }
    return pack_pair_classification(sentence1, sentence2, labels), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=None, help="Path to JuRo root directory")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    source_dir = _resolve_source_dir(args.source_dir)
    print(f"Using JuRo source: {source_dir}")

    test_ds, stats = _build_pair_classification(source_dir)
    print_pair_stats(
        "JuRo PairClassification",
        stats,
        test_ds["sentence1"],
        test_ds["sentence2"],
        test_ds["label"],
    )

    if args.dry_run:
        return

    out = DatasetDict({"test": test_ds})
    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message="RoMTEB JuRo legal MCQ (PairClassification, one row per pair)",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
