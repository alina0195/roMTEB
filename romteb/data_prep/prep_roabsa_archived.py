"""ARCHIVED — document-level RoABSA. Replaced by aspect-level prep_roabsa.py.

Prepare and upload RoABSA for RoMTEB (document-level classification).

Source: upb-nlp/RoABSA — Romanian aspect-based sentiment on product reviews.

Each review becomes one classification example:
  - text: the review ``body``
  - label: majority polarity parsed from ``aspects_polarities``

Polarities are taken from each ``aspect - polarity`` pair, e.g.
``"product - negative"`` -> ``negative``. When multiple aspects are present
(``{"choices": ["product - negative", "quality - positive", ...]}``), we
collect every polarity and use the most frequent one as the document label.

Output: alina0195/romteb-roabsa with native train/test splits and columns
        {text, label, label_name}.
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
    """Return raw ``aspect - polarity`` strings from one cell."""
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


def _parse_polarity(entry: str) -> str | None:
    """Extract polarity token from ``aspect - polarity``."""
    m = _PAIR_RE.match(entry.strip())
    if not m:
        return None
    token = m.group(2).strip().lower()
    if token in POLARITY_MAP:
        return LABEL_NAMES[POLARITY_MAP[token]]
    return None


def _majority_polarity(raw) -> str | None:
    polarities = []
    for entry in _aspect_entries(raw):
        pol = _parse_polarity(entry)
        if pol:
            polarities.append(pol)
    if not polarities:
        return None
    counts = Counter(polarities)
    # Tie-break: negative > neutral > positive (more conservative document label).
    return max(counts.items(), key=lambda kv: (kv[1], {"negative": 2, "neutral": 1, "positive": 0}[kv[0]]))[0]


def _row_to_example(item: dict) -> dict | None:
    body = (item.get("body") or item.get("text") or "").strip()
    if not body:
        return None
    label_name = _majority_polarity(item.get("aspects_polarities"))
    if label_name is None:
        return None
    label = POLARITY_MAP[label_name]
    return {"text": body, "label": label, "label_name": label_name}


def _build_split(records: list[dict]) -> Dataset:
    rows = []
    for item in records:
        ex = _row_to_example(item)
        if ex:
            rows.append(ex)
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Loading {SOURCE_REPO} via parquet (Json feature workaround) ...")
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
            f"classification rows"
        )
        print(f"    label distribution: {dict(Counter(ds['label_name']))}")
        out[split_name] = ds

    print_stats("RoABSA (document-level majority polarity)", out, text_keys=["text"])

    if args.dry_run:
        return

    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message=(
            "RoMTEB RoABSA document-level sentiment "
            "(majority polarity from aspects_polarities)"
        ),
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
