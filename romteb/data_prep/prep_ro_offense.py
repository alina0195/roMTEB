"""Prepare and upload Ro-Offense for RoMTEB.

Source: readerbench/news-ro-offense - 5-class offense classification on
Romanian news comments. Labels in the source are typically strings; we
normalize them to ints.

Output: alina0195/romteb-ro-offense with {text, label} columns.
"""

from __future__ import annotations

import argparse

from romteb._hf_datasets import Dataset, DatasetDict, load_dataset

from romteb.data_prep._common import print_stats, push_to_hub
from romteb.data_prep._classification_utils import (
    ensure_int_labels,
    stratified_train_test_split,
)


SOURCE_REPO = "readerbench/news-ro-offense"
TARGET_REPO = "alina0195/romteb-ro-offense"


TEXT_KEYS = ("comment_text", "text", "comment", "content", "body")
LABEL_KEYS = ("LABEL", "label", "labels", "offense", "class", "category")


def _pick(cols: list[str], candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of {candidates} found in {cols}")


def _normalize(ds: Dataset) -> Dataset:
    cols = ds.column_names
    text_col = _pick(cols, TEXT_KEYS)
    label_col = _pick(cols, LABEL_KEYS)
    ds = ds.map(
        lambda r: {"text": (r[text_col] or "").strip(), "label": r[label_col]},
        remove_columns=[c for c in cols if c not in {"text", "label"}],
    )
    ds = ds.filter(lambda r: r["text"] and r["label"] is not None)
    ds, _ = ensure_int_labels(ds)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Loading {SOURCE_REPO} ...")
    raw = load_dataset(SOURCE_REPO)

    if isinstance(raw, DatasetDict) and "train" in raw and "test" in raw:
        out = DatasetDict(train=_normalize(raw["train"]), test=_normalize(raw["test"]))
    else:
        base = raw[list(raw.keys())[0]] if isinstance(raw, DatasetDict) else raw
        out = stratified_train_test_split(_normalize(base))

    print_stats("Ro-Offense", out, text_keys=["text"])

    if args.dry_run:
        return
    sha = push_to_hub(out, TARGET_REPO, commit_message="Initial RoMTEB Ro-Offense (5-class)")
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
