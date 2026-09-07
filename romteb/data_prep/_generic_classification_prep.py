"""Generic prep flow used by the Tier-2 classification datasets.

Each Tier-2 prep script provides:
  - CANDIDATE_SOURCES: HF dataset paths to try, in priority order
  - TEXT_KEYS, LABEL_KEYS: candidate column names
  - TARGET_REPO: where to upload the normalized dataset

Then calls `run_generic_prep(...)` from this module.
"""

from __future__ import annotations

from romteb._hf_datasets import Dataset, DatasetDict, load_dataset

from romteb.data_prep._common import print_stats, push_to_hub
from romteb.data_prep._classification_utils import (
    ensure_int_labels,
    stratified_train_test_split,
)


def _load_first_available(candidates: list[str]):
    for src in candidates:
        try:
            print(f"Trying {src} ...")
            return src, load_dataset(src)
        except Exception as exc:
            print(f"  {src}: {exc}")
    return None, None


def _pick(cols: list[str], cands: tuple[str, ...]) -> str | None:
    for c in cands:
        if c in cols:
            return c
    return None


def _normalize(
    ds: Dataset, text_keys: tuple[str, ...], label_keys: tuple[str, ...]
) -> Dataset:
    cols = ds.column_names
    text_col = _pick(cols, text_keys)
    label_col = _pick(cols, label_keys)
    if text_col is None or label_col is None:
        raise SystemExit(
            f"Cannot map columns: text in {text_keys}, label in {label_keys}; "
            f"found {cols}"
        )
    ds = ds.map(
        lambda r: {"text": (r[text_col] or "").strip(), "label": r[label_col]},
        remove_columns=[c for c in cols if c not in {"text", "label"}],
    )
    ds = ds.filter(lambda r: r["text"] and r["label"] is not None)
    ds, _ = ensure_int_labels(ds)
    return ds


def run_generic_prep(
    name: str,
    candidate_sources: list[str],
    target_repo: str,
    text_keys: tuple[str, ...] = ("text", "content", "body", "comment", "tweet"),
    label_keys: tuple[str, ...] = ("label", "labels", "class", "category"),
    dry_run: bool = False,
    on_missing: str = "skip",
) -> str | None:
    """Generic prep pipeline. Returns the upload SHA or None when skipped."""
    src, raw = _load_first_available(candidate_sources)
    if raw is None:
        msg = (
            f"[{name}] no source available from {candidate_sources}. "
            f"This dataset will be dropped from v1; document in _inventory.md."
        )
        if on_missing == "raise":
            raise SystemExit(msg)
        print(msg)
        return None

    print(f"Loaded {src}; splits: {list(raw.keys()) if isinstance(raw, DatasetDict) else 'flat'}")

    if isinstance(raw, DatasetDict) and "train" in raw and "test" in raw:
        out = DatasetDict(
            train=_normalize(raw["train"], text_keys, label_keys),
            test=_normalize(raw["test"], text_keys, label_keys),
        )
    else:
        base = raw[list(raw.keys())[0]] if isinstance(raw, DatasetDict) else raw
        out = stratified_train_test_split(_normalize(base, text_keys, label_keys))

    print_stats(name, out, text_keys=["text"])

    if dry_run:
        return None
    sha = push_to_hub(out, target_repo, commit_message=f"Initial RoMTEB {name}")
    print(f"\nPin: revision = {sha!r}")
    return sha
