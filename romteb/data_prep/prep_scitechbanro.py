"""Prepare SciTechBanRO from the ClickbaitSciTechRO GitHub xlsx.

Source: https://github.com/ralucaginga/ClickbaitSciTechRO/blob/main/dataset.xlsx
Output: alina0195/romteb-scitechbanro  {text, label, label_name}
"""

from __future__ import annotations

import argparse
import io
from urllib.request import Request, urlopen

import pandas as pd
from romteb._hf_datasets import Dataset

from romteb.data_prep._classification_utils import (
    ensure_int_labels,
    stratified_train_test_split,
)
from romteb.data_prep._common import print_stats, push_to_hub


XLSX_URL = (
    "https://github.com/ralucaginga/ClickbaitSciTechRO/raw/main/dataset.xlsx"
)
TARGET_REPO = "alina0195/romteb-scitechbanro"

TEXT_CANDIDATES = (
    "text",
    "content",
    "body",
    "article",
    "title",
    "headline",
    "news",
    "document",
)
LABEL_CANDIDATES = (
    "label",
    "class",
    "category",
    "topic",
    "domain",
    "clickbait",
    "target",
    "y",
)


def _download_xlsx(url: str) -> pd.DataFrame:
    print(f"Downloading {url} ...")
    req = Request(url, headers={"User-Agent": "romteb-prep/1.0"})
    with urlopen(req, timeout=120) as resp:
        data = resp.read()
    return pd.read_excel(io.BytesIO(data), engine="openpyxl")


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _combine_text(row: pd.Series, text_col: str, extra_cols: list[str]) -> str:
    parts = []
    title = None
    for col in extra_cols:
        if col.lower() in {"title", "headline"} and col != text_col:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                title = val.strip()
    body = row.get(text_col)
    body_s = ("" if body is None or (isinstance(body, float) and pd.isna(body)) else str(body)).strip()
    if title and title not in body_s:
        parts.append(title)
    if body_s:
        parts.append(body_s)
    return "\n".join(parts).strip()


def _normalize(df: pd.DataFrame) -> Dataset:
    cols = [str(c) for c in df.columns]
    print(f"  columns: {cols}  rows={len(df):,}")
    text_col = _pick(cols, TEXT_CANDIDATES)
    label_col = _pick(cols, LABEL_CANDIDATES)
    if text_col is None:
        # Fallback: longest string-like column.
        str_cols = [c for c in cols if df[c].dtype == object]
        if not str_cols:
            raise SystemExit(f"No text column in {cols}")
        text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
        print(f"  inferred text column: {text_col}")
    if label_col is None:
        raise SystemExit(f"Cannot find a label column in {cols}")
    print(f"  using text={text_col!r}  label={label_col!r}")

    rows = []
    for _, row in df.iterrows():
        text = _combine_text(row, text_col, cols)
        label = row.get(label_col)
        if not text or label is None or (isinstance(label, float) and pd.isna(label)):
            continue
        if isinstance(label, str):
            label = label.strip()
            if not label:
                continue
        rows.append({"text": text, "label": label})
    if not rows:
        raise SystemExit("No usable SciTechBanRO rows after filtering.")
    ds = Dataset.from_list(rows)
    ds, id2label = ensure_int_labels(ds)
    if id2label:
        ds = ds.map(lambda r: {"label_name": id2label.get(int(r["label"]), str(r["label"]))})
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--xlsx_url", default=XLSX_URL)
    args = parser.parse_args()

    df = _download_xlsx(args.xlsx_url)
    ds = _normalize(df)
    out = stratified_train_test_split(ds)
    print_stats("SciTechBanRO", out, text_keys=["text"])

    if args.dry_run:
        return
    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message="RoMTEB SciTechBanRO from ClickbaitSciTechRO dataset.xlsx",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
