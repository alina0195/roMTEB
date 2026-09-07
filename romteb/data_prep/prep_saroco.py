"""Prepare SaRoCo (Romanian sarcasm) from GitHub CSVs.

Source: https://github.com/MihaelaGaman/SaRoCo/tree/master/data
Output: alina0195/romteb-saroco  {text, label, label_name}
"""

from __future__ import annotations

import argparse
from urllib.request import Request, urlopen

import pandas as pd
from romteb._hf_datasets import Dataset, DatasetDict

from romteb.data_prep._classification_utils import ensure_int_labels
from romteb.data_prep._common import print_stats, push_to_hub


BASE = "https://raw.githubusercontent.com/MihaelaGaman/SaRoCo/master/data"
SPLIT_FILES = {
    "train": f"{BASE}/train.csv",
    "validation": f"{BASE}/validation.csv",
    "test": f"{BASE}/test.csv",
}
TARGET_REPO = "alina0195/romteb-saroco"

TEXT_CANDIDATES = (
    "text",
    "tweet",
    "content",
    "body",
    "sentence",
    "message",
    "document",
    "article",
    "news",
    "title",
)
LABEL_CANDIDATES = ("label", "sarcasm", "class", "target", "y")


def _download_csv(url: str) -> pd.DataFrame:
    print(f"Downloading {url} ...")
    req = Request(url, headers={"User-Agent": "romteb-prep/1.0"})
    with urlopen(req, timeout=120) as resp:
        return pd.read_csv(resp)


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _normalize(df: pd.DataFrame) -> Dataset:
    cols = [str(c) for c in df.columns]
    print(f"  columns: {cols}  rows={len(df):,}")
    text_col = _pick(cols, TEXT_CANDIDATES)
    label_col = _pick(cols, LABEL_CANDIDATES)
    if text_col is None:
        str_cols = [c for c in cols if df[c].dtype == object]
        text_col = str_cols[0] if str_cols else None
    if text_col is None or label_col is None:
        raise SystemExit(f"Cannot map SaRoCo columns {cols} to text/label")
    print(f"  using text={text_col!r}  label={label_col!r}")

    rows = []
    for _, row in df.iterrows():
        raw = row.get(text_col)
        text = ("" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw)).strip()
        label = row.get(label_col)
        if not text or label is None or (isinstance(label, float) and pd.isna(label)):
            continue
        if isinstance(label, str):
            label = label.strip()
            if not label:
                continue
        rows.append({"text": text, "label": label})
    ds = Dataset.from_list(rows)
    ds, id2label = ensure_int_labels(ds)
    if id2label:
        ds = ds.map(lambda r: {"label_name": id2label.get(int(r["label"]), str(r["label"]))})
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    splits = {}
    for name, url in SPLIT_FILES.items():
        df = _download_csv(url)
        splits[name] = _normalize(df)

    # MTEB classification trains on `train`, evaluates on `test`.
    out = DatasetDict(train=splits["train"], test=splits["test"])
    if "validation" in splits:
        print(f"  (validation split loaded: {len(splits['validation']):,} rows, not used by MTEB probe)")
    print_stats("SaRoCo", out, text_keys=["text"])

    if args.dry_run:
        return
    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message="RoMTEB SaRoCo sarcasm classification from GitHub CSVs",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
