"""Prepare REDv2 as a single-label classification dataset for RoMTEB.

REDv2 is multi-label. RoMTEB classification tasks currently rely on
single-label `AbsTaskClassification`, so we project each example to one
"dominant" label by taking the unique argmax of `sum_labels`.
Tied maxima are dropped to avoid ambiguous supervision.
"""

from __future__ import annotations

import argparse

from romteb._hf_datasets import DatasetDict, load_dataset

from romteb.data_prep._common import print_stats, push_to_hub


TARGET_REPO = "alina0195/romteb-red-v2-emotion"
RED_LABELS = [
    "sadness",
    "surprise",
    "fear",
    "anger",
    "neutral",
    "trust",
    "joy",
]
SOURCE_FILES = {
    "train": "https://raw.githubusercontent.com/Alegzandra/RED-Romanian-Emotion-Datasets/main/REDv2/data/train.json",
    "validation": "https://raw.githubusercontent.com/Alegzandra/RED-Romanian-Emotion-Datasets/main/REDv2/data/valid.json",
    "test": "https://raw.githubusercontent.com/Alegzandra/RED-Romanian-Emotion-Datasets/main/REDv2/data/test.json",
}


def _dominant_label(sum_labels: list[int]) -> int | None:
    if not sum_labels:
        return None
    max_val = max(sum_labels)
    if max_val <= 0:
        return None
    winners = [i for i, v in enumerate(sum_labels) if v == max_val]
    if len(winners) != 1:
        return None
    return winners[0]


def _normalize_split(split):
    def _map_row(row):
        text = (row.get("text") or "").strip()
        label_idx = _dominant_label(list(row.get("sum_labels") or []))
        label_name = RED_LABELS[label_idx] if label_idx is not None else None
        return {"text": text, "label": label_idx, "label_name": label_name}

    mapped = split.map(_map_row)
    mapped = mapped.filter(lambda r: bool(r["text"]) and r["label"] is not None)
    keep_cols = {"text", "label", "label_name"}
    drop_cols = [c for c in mapped.column_names if c not in keep_cols]
    if drop_cols:
        mapped = mapped.remove_columns(drop_cols)
    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    raw = load_dataset("json", data_files=SOURCE_FILES)
    out = DatasetDict({split: _normalize_split(ds) for split, ds in raw.items()})

    print_stats("REDv2EmotionClassification", out, text_keys=["text"])
    print(f"Labels: {dict(enumerate(RED_LABELS))}")

    if args.dry_run:
        return
    sha = push_to_hub(out, TARGET_REPO, commit_message="Initial RoMTEB REDv2 upload")
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
