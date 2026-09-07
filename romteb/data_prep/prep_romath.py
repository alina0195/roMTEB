"""Prepare RoMath as a domain-classification dataset for RoMTEB.

Source: ``cosmadrian/romath`` (gated, CC-BY-NC-4.0).
Three subsets: RoMath-Baccalaureate, RoMath-Competitions, RoMath-Synthetic.
Each row has a ``question`` (math problem in Romanian) and a ``domain``
(mathematical area like "algebra", "geometry", etc.).

The prep merges all subsets, uses the ``domain`` column as the classification
label, and does an 80/20 stratified split when no pre-defined test split
exists.
"""

from __future__ import annotations

import argparse

from romteb._hf_datasets import DatasetDict, concatenate_datasets, load_dataset

from romteb.data_prep._classification_utils import (
    ensure_int_labels,
    stratified_train_test_split,
)
from romteb.data_prep._common import print_stats, push_to_hub

SOURCE_REPO = "cosmadrian/romath"
TARGET_REPO = "alina0195/romteb-romath-domain"

SUBSETS = [
    "bac",
    "comps",
    "synthetic",
]

TEXT_KEYS = ("question", "problem", "text")
LABEL_KEYS = ("domain", "category", "subject")


def _pick(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _load_and_merge() -> DatasetDict | None:
    """Load all subsets and merge into a single Dataset per split."""
    all_train, all_test = [], []

    for subset in SUBSETS:
        try:
            raw = load_dataset(SOURCE_REPO, subset)
        except Exception as exc:
            print(f"  {subset}: {exc}")
            continue

        if isinstance(raw, DatasetDict):
            splits = list(raw.keys())
            print(f"  {subset}: splits={splits}, rows={sum(len(raw[s]) for s in splits):,}")
            if "test" in raw:
                all_test.append(raw["test"])
            if "train" in raw:
                all_train.append(raw["train"])
            elif "test" not in raw:
                all_train.append(raw[splits[0]])
        else:
            print(f"  {subset}: flat, rows={len(raw):,}")
            all_train.append(raw)

    if not all_train and not all_test:
        return None

    merged_train = concatenate_datasets(all_train) if all_train else None
    merged_test = concatenate_datasets(all_test) if all_test else None

    if merged_train is not None and merged_test is not None:
        return DatasetDict(train=merged_train, test=merged_test)
    combined = merged_train or merged_test
    return DatasetDict(all=combined)


def _normalize(ds, text_col: str, label_col: str):
    ds = ds.map(
        lambda r: {"text": (r[text_col] or "").strip(), "label": r[label_col]},
        remove_columns=[c for c in ds.column_names if c not in {"text", "label"}],
    )
    ds = ds.filter(lambda r: bool(r["text"]) and r["label"] is not None and str(r["label"]).strip() != "")
    ds, id2label = ensure_int_labels(ds)
    return ds, id2label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Loading {SOURCE_REPO} (gated — access required) ...")
    merged = _load_and_merge()
    if merged is None:
        raise SystemExit("No data loaded. Check HF access and subset names.")

    first_split = merged[list(merged.keys())[0]]
    cols = first_split.column_names
    print(f"Merged columns: {cols}")

    text_col = _pick(cols, TEXT_KEYS)
    label_col = _pick(cols, LABEL_KEYS)
    if text_col is None or label_col is None:
        raise SystemExit(
            f"Cannot map columns: text in {TEXT_KEYS}, label in {LABEL_KEYS}; found {cols}"
        )
    print(f"Using text_col={text_col!r}, label_col={label_col!r}")

    if "train" in merged and "test" in merged:
        train_ds, id2label = _normalize(merged["train"], text_col, label_col)
        test_ds, _ = _normalize(merged["test"], text_col, label_col)
        out = DatasetDict(train=train_ds, test=test_ds)
    else:
        base = merged[list(merged.keys())[0]]
        norm, id2label = _normalize(base, text_col, label_col)
        out = stratified_train_test_split(norm)

    print_stats("RoMathDomainClassification", out, text_keys=["text"])
    print(f"Domain labels: {id2label}")

    if args.dry_run:
        return
    sha = push_to_hub(out, TARGET_REPO, commit_message="RoMTEB RoMath domain classification")
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
