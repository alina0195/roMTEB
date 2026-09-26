"""Prepare and upload RoNLI for RoMTEB.

Source: https://github.com/Eduard6421/RONLI (Poesina et al., ACL 2024),
pinned to commit ``RONLI_COMMIT``. Only these three files are read:

    dataset/datasets/train.json        55,102 pairs, distant supervision
    dataset/datasets/validation.json    3,059 pairs, manually annotated
    dataset/datasets/test.json          3,000 pairs, manually annotated

The repo also ships train_easy/train_hard/train_curriculum*/old_validation
and a second validation.json; they must NOT be globbed into the splits.

Upstream label ids, confirmed in two places of the RONLI repo:
corpus/NLI_extractor.py (label assignment) and
dataset/generate_datasets/check_outputs.py (word_to_class):

    0 = Contrastive  (contradiction; markers such as "În contrast", "Contrar ...")
    1 = Entailment
    2 = Consequence  ("reasoning"; markers such as "prin urmare", "astfel")
    3 = Unrelated    (random pairs of unrelated sentences; the "neutral" class)

The previous version of this script assumed 0=entailment, 1=neutral,
2=contradiction and silently dropped id 3, so the pair-classification task
compared Contrastive (as "positive") against Consequence. See the task file.

Output: alina0195/romteb-ronli, columns
    sentence1, sentence2, label (upstream id 0-3), label_name, guid, source_split

    train : upstream train.json verbatim (for training only; never evaluated)
    test  : upstream validation.json + test.json, exact-duplicate pairs merged,
            pairs whose label differs between the two files dropped.
            No rebalancing. There is no validation split: it is part of test,
            so nothing may be tuned or early-stopped on it.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

RONLI_COMMIT = "fd75ce0a8cf09af638a94efea2e0e016bbfcb112"
GITHUB_ZIP_URL = f"https://github.com/Eduard6421/RONLI/archive/{RONLI_COMMIT}.zip"
SPLIT_FILES = {
    "train": "dataset/datasets/train.json",
    "validation": "dataset/datasets/validation.json",
    "test": "dataset/datasets/test.json",
}
EXPECTED_ROWS = {"train": 55_102, "validation": 3_059, "test": 3_000}

TARGET_REPO = "alina0195/romteb-ronli"

# Names exactly as upstream word_to_class (check_outputs.py).
LABEL_NAMES = {0: "Contrastive", 1: "Entailment", 2: "Consequence", 3: "Unrelated"}


# --------------------------------------------------------------------------
# Loading (pure Python, no HF dependency, so it can be unit-tested)
# --------------------------------------------------------------------------

def load_raw_splits(source_dir: str | None = None) -> dict[str, list[dict]]:
    """Read the three split files from a local clone or the pinned GitHub zip."""
    if source_dir:
        root = Path(source_dir)
        return {
            split: json.loads((root / rel).read_text(encoding="utf-8"))
            for split, rel in SPLIT_FILES.items()
        }

    print(f"Downloading {GITHUB_ZIP_URL} ...")
    with urllib.request.urlopen(GITHUB_ZIP_URL, timeout=300) as resp:
        blob = resp.read()
    out: dict[str, list[dict]] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        prefix = z.namelist()[0].split("/")[0]  # "RONLI-<sha>"
        for split, rel in SPLIT_FILES.items():
            with z.open(f"{prefix}/{rel}") as f:
                out[split] = json.load(f)
    return out


def normalize(records: list[dict], split: str) -> list[dict]:
    """Keep upstream ids as they are; fail loudly on anything unexpected."""
    rows = []
    for r in records:
        s1 = (r.get("sentence1") or "").strip()
        s2 = (r.get("sentence2") or "").strip()
        label = r.get("label")
        if label not in LABEL_NAMES:
            raise ValueError(f"[{split}] unexpected label {label!r} in guid={r.get('guid')}")
        if not s1 or not s2:
            continue
        rows.append({
            "sentence1": s1,
            "sentence2": s2,
            "label": int(label),
            "label_name": LABEL_NAMES[label],
            "guid": str(r.get("guid", "")),
            "source_split": split,
        })
    return rows


def merge_eval_splits(val_rows: list[dict], test_rows: list[dict]) -> tuple[list[dict], dict]:
    """Union of validation and test, deduplicated on the (sentence1, sentence2) pair.

    Same pair + same label in both files -> kept once (source_split="test+validation").
    Same pair + different labels         -> dropped (annotation conflict).
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in test_rows + val_rows:  # test first, so its guid wins on duplicates
        groups.setdefault((r["sentence1"], r["sentence2"]), []).append(r)

    merged, n_dup, n_conflict = [], 0, 0
    for rows in groups.values():
        labels = {r["label"] for r in rows}
        if len(labels) > 1:
            n_conflict += 1
            continue
        keep = dict(rows[0])
        if len(rows) > 1:
            n_dup += 1
            keep["source_split"] = "+".join(sorted({r["source_split"] for r in rows}))
        merged.append(keep)
    return merged, {"duplicates_merged": n_dup, "conflicts_dropped": n_conflict}


def leakage_report(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    train_pairs = {(r["sentence1"], r["sentence2"]) for r in train_rows}
    train_sents = {r["sentence1"] for r in train_rows} | {r["sentence2"] for r in train_rows}
    return {
        "eval_pairs_also_in_train": sum((r["sentence1"], r["sentence2"]) in train_pairs for r in eval_rows),
        "eval_pairs_sharing_a_sentence_with_train": sum(
            r["sentence1"] in train_sents or r["sentence2"] in train_sents for r in eval_rows
        ),
    }


def build(source_dir: str | None = None) -> tuple[dict[str, list[dict]], dict]:
    raw = load_raw_splits(source_dir)
    for split, n in EXPECTED_ROWS.items():
        if len(raw[split]) != n:
            print(f"[ronli] WARNING: {split} has {len(raw[split])} rows, expected {n}")

    norm = {split: normalize(recs, split) for split, recs in raw.items()}
    test, merge_stats = merge_eval_splits(norm["validation"], norm["test"])
    if merge_stats["conflicts_dropped"] == 0 and merge_stats["duplicates_merged"] == 0:
        print("[ronli] note: validation and test share no pairs")

    stats = {
        "label_counts": {
            split: dict(Counter(r["label_name"] for r in rows))
            for split, rows in {"train": norm["train"], "test": test}.items()
        },
        "binary_test_pairs(Entailment vs Contrastive)": dict(
            Counter(r["label_name"] for r in test if r["label"] in (0, 1))
        ),
        **merge_stats,
        **leakage_report(norm["train"], test),
    }
    return {"train": norm["train"], "test": test}, stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Build and print stats, do not push.")
    parser.add_argument("--source_dir", default=None, help="Local RONLI clone instead of the GitHub zip.")
    args = parser.parse_args()

    splits, stats = build(args.source_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    # HF imports only when pushing (they go through the local-datasets-folder shim).
    from romteb._hf_datasets import Dataset, DatasetDict
    from romteb.data_prep._common import print_stats, push_to_hub

    out = DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()})
    print_stats("RoNLI", out, text_keys=["sentence1", "sentence2"])
    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message=(
            f"RoNLI from RONLI@{RONLI_COMMIT[:7]}: correct upstream label ids "
            "(0 Contrastive, 1 Entailment, 2 Consequence, 3 Unrelated); "
            "test = validation+test deduplicated; no rebalancing"
        ),
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()