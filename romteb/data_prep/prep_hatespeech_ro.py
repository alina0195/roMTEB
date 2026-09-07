"""Prepare HateSpeech-RO (binary: hate vs. non-hate).

Source: https://github.com/andra-pumnea/hate-speech-ro
CSV file: hate-speech-labeled-corrected.csv (semicolon-delimited,
columns: index, text, label where label 0=non-hate, 1=hate).

~2500 Romanian comments from social media about the 2018 constitutional
referendum on same-sex marriage.
"""

from __future__ import annotations

import argparse
import io
import urllib.request

import pandas as pd
from romteb._hf_datasets import Dataset, DatasetDict

from romteb.data_prep._common import print_stats, push_to_hub
from romteb.data_prep._classification_utils import stratified_train_test_split


GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/andra-pumnea/hate-speech-ro"
    "/master/hate-speech-labeled-corrected.csv"
)
TARGET_REPO = "alina0195/romteb-hatespeech-ro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Downloading {GITHUB_CSV_URL} ...")
    with urllib.request.urlopen(GITHUB_CSV_URL, timeout=120) as resp:
        raw = resp.read()

    df = pd.read_csv(io.BytesIO(raw), sep=";", index_col=0)
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0]
    print(f"After cleaning: {len(df):,} rows")
    print(f"Label distribution: {dict(df['label'].value_counts().sort_index())}")

    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    out = stratified_train_test_split(ds)

    print_stats("HateSpeech-RO", out, text_keys=["text"])

    if args.dry_run:
        return
    sha = push_to_hub(
        out, TARGET_REPO,
        commit_message="RoMTEB HateSpeech-RO (binary, from GitHub CSV)",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
