"""RoABSA aspect-level sentiment: encode review text together with the aspect.

Each ``aspect - polarity`` pair becomes one classification example. The
string passed to the frozen embedder is:

    Entitate: {aspect}
    {review body}

so the logistic-regression probe predicts polarity *towards that entity*,
not document-level majority sentiment.

Source: upb-nlp/RoABSA parquet (Json feature workaround).
Output: alina0195/romteb-roabsa  {text, label, label_name, aspect}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import pandas as pd
from romteb._hf_datasets import Dataset, DatasetDict

from romteb.data_prep._common import print_stats, push_to_hub


SOURCE_REPO = "upb-nlp/RoABSA"
TARGET_REPO = "alina0195/romteb-roabsa"

PARQUET_URLS = {
    "train": (
        "https://huggingface.co/datasets/upb-nlp/RoABSA/resolve/"
        "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    ),
    "test": (
        "https://huggingface.co/datasets/upb-nlp/RoABSA/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
    ),
}

POLARITY_MAP = {
    "negative": 0,
    "neg": 0,
    "neutral": 1,
    "neu": 1,
    "positive": 2,
    "pos": 2,
}

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

_PAIR_RE = re.compile(
    r"^(.+?)\s*-\s*(negative|neg|neutral|neu|positive|pos)$",
    re.IGNORECASE,
)


def _normalize_aspects_polarities(raw) -> dict | str | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
    return None


def _aspect_entries(raw) -> list[str]:
    ap = _normalize_aspects_polarities(raw)
    if ap is None:
        return []
    if isinstance(ap, str):
        return [ap]
    if isinstance(ap, dict):
        choices = ap.get("choices") or []
        return [c for c in choices if isinstance(c, str) and c.strip()]
    if isinstance(ap, list):
        return [c for c in ap if isinstance(c, str) and c.strip()]
    return []


def _parse_pair(entry: str) -> tuple[str, str] | None:
    m = _PAIR_RE.match(entry.strip())
    if not m:
        return None
    aspect = m.group(1).strip()
    token = m.group(2).strip().lower()
    if token not in POLARITY_MAP or not aspect:
        return None
    return aspect, LABEL_NAMES[POLARITY_MAP[token]]


def _encode_text(body: str, aspect: str) -> str:
    return f"Entitate: {aspect}\n{body}"


def _row_to_examples(item: dict) -> list[dict]:
    body = (item.get("body") or item.get("text") or "").strip()
    if not body:
        return []
    out = []
    for entry in _aspect_entries(item.get("aspects_polarities")):
        parsed = _parse_pair(entry)
        if parsed is None:
            continue
        aspect, label_name = parsed
        out.append(
            {
                "text": _encode_text(body, aspect),
                "label": POLARITY_MAP[label_name],
                "label_name": label_name,
                "aspect": aspect,
            }
        )
    return out


def _build_split(records: list[dict]) -> Dataset:
    rows = []
    for item in records:
        rows.extend(_row_to_examples(item))
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Loading {SOURCE_REPO} via parquet (aspect-level) ...")
    split_dfs: dict[str, pd.DataFrame] = {}
    for split_name, url in PARQUET_URLS.items():
        print(f"  Downloading {split_name} from {url} ...")
        split_dfs[split_name] = pd.read_parquet(url)

    out = DatasetDict()
    for split_name, df in split_dfs.items():
        records = df.to_dict(orient="records")
        ds = _build_split(records)
        print(
            f"  {split_name}: {len(df):,} reviews -> {len(ds):,} "
            f"aspect-level rows"
        )
        print(f"    label distribution: {dict(Counter(ds['label_name']))}")
        out[split_name] = ds

    print_stats("RoABSA (aspect-level: entity + review)", out, text_keys=["text"])

    if args.dry_run:
        return

    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message=(
            "RoMTEB RoABSA aspect-level sentiment "
            "(encode 'Entitate: {aspect}\\n{body}')"
        ),
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
