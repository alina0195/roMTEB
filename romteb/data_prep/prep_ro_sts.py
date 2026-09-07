"""Prepare and upload RO-STS for RoMTEB.

Source: dumitrescustefan/ro_sts (8628 sentence pairs, scores 0-5).
Output: alina0195/romteb-ro-sts with three splits (train/validation/test)
        and columns {sentence1, sentence2, score}.

USAGE:
    ./apptainer-exec-romteb.sh romteb/data_prep/prep_ro_sts.py
    ./apptainer-exec-romteb.sh romteb/data_prep/prep_ro_sts.py --dry_run
"""

from __future__ import annotations

import argparse

from urllib.request import urlopen

from romteb._hf_datasets import Dataset, DatasetDict

from romteb.data_prep._common import print_stats, push_to_hub


SOURCE_REPO = "dumitrescustefan/ro_sts"
TARGET_REPO = "alina0195/romteb-ro-sts"
# HF `dumitrescustefan/ro_sts` is a legacy dataset *script* (no parquet).
# Current `datasets` refuses scripts; the TSVs are the canonical LiRo files.
_TSV_BASE = (
    "https://raw.githubusercontent.com/dumitrescustefan/RO-STS/"
    "master/dataset/text-similarity/"
)
_SPLIT_FILES = {
    "train": "RO-STS.train.tsv",
    "validation": "RO-STS.dev.tsv",
    "test": "RO-STS.test.tsv",
}


def _load_tsv_split(filename: str) -> Dataset:
    url = _TSV_BASE + filename
    print(f"  fetching {url}")
    with urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    rows: list[dict] = []
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        score = float(parts[0])
        s1, s2 = parts[1].strip(), parts[2].strip()
        if s1 and s2 and 0.0 <= score <= 5.0:
            rows.append({"sentence1": s1, "sentence2": s2, "score": score})
    if not rows:
        raise RuntimeError(f"no valid rows in {filename}")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Skip the HF push step")
    args = parser.parse_args()

    print(f"Loading RO-STS TSVs (same files as {SOURCE_REPO}) ...")
    normalized = DatasetDict(
        {split: _load_tsv_split(name) for split, name in _SPLIT_FILES.items()}
    )
    print_stats("RO-STS", normalized, text_keys=["sentence1", "sentence2"])

    if args.dry_run:
        print("\n--dry_run set; not pushing to HF.")
        return

    sha = push_to_hub(
        normalized,
        TARGET_REPO,
        commit_message="Initial RoMTEB RO-STS (8628 pairs, score in [0,5])",
    )
    print(f"\nPin this in romteb/tasks/sts/ro_sts.py: revision = {sha!r}")


if __name__ == "__main__":
    main()
