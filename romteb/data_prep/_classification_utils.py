"""Shared helpers for classification dataset prep scripts."""

from __future__ import annotations

from romteb._hf_datasets import Dataset, DatasetDict


def ensure_int_labels(ds: Dataset, label_col: str = "label") -> tuple[Dataset, dict[int, str]]:
    """Convert string labels to ints if needed; returns (ds, id2label)."""
    if not ds.column_names:
        return ds, {}
    sample = ds[label_col][0] if label_col in ds.column_names else None
    if isinstance(sample, int):
        try:
            id2label = {i: name for i, name in enumerate(ds.features[label_col].names)}
        except (AttributeError, KeyError):
            id2label = {}
        return ds, id2label
    classes = sorted({x for x in ds[label_col] if x is not None})
    label2id = {c: i for i, c in enumerate(classes)}
    ds = ds.map(lambda r: {label_col: label2id[r[label_col]]})
    return ds, {i: c for c, i in label2id.items()}


def stratified_train_test_split(
    ds: Dataset,
    test_size: float = 0.2,
    seed: int = 42,
    label_col: str = "label",
) -> DatasetDict:
    """80/20 stratified split. Falls back to random if stratification fails."""
    try:
        split = ds.train_test_split(
            test_size=test_size, seed=seed, stratify_by_column=label_col
        )
    except (ValueError, KeyError):
        split = ds.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict(train=split["train"], test=split["test"])
