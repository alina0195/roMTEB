"""Prepare and upload RoNLI for RoMTEB.

Source: https://github.com/Eduard6421/RONLI (~64K Romanian NLI triples,
3-class: entailment, neutral, contradiction).

Tries HF mirror first, falls back to downloading raw files from GitHub.

Output: alina0195/romteb-ronli with {sentence1, sentence2, label} columns,
        label in {0=entailment, 1=neutral, 2=contradiction}.

Test split is rebalanced by undersampling contradiction so that entailment
prevalence is ~50% after the ``dataset_transform`` filter that drops neutral.
The upstream RONLI test set is 92.8% contradiction / 7.2% entailment; that
prevalence is below the max_ap floor of a random classifier and produced a
0.10-0.14 spread across all 13 models in the roMTEB v1 legacy run.
Train and validation splits are preserved verbatim.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
import zipfile
from typing import Iterable

from romteb._hf_datasets import Dataset, DatasetDict

from romteb.data_prep._common import print_stats, push_to_hub

REBALANCE_SEED = 20260901
REBALANCE_NEG_PER_POS = 1  # positives : negatives ratio in the rebuilt test split


TARGET_REPO = "alina0195/romteb-ronli"

GITHUB_ZIP_URL = "https://github.com/Eduard6421/RONLI/archive/refs/heads/main.zip"

HF_CANDIDATES = [
    "Eduard6421/RONLI",
    "readerbench/RONLI",
    "ronli",
]

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "0": 0,
    "1": 1,
    "2": 2,
    0: 0,
    1: 1,
    2: 2,
}

SENT1_KEYS = ("premise", "sentence1", "anchor", "text_a")
SENT2_KEYS = ("hypothesis", "sentence2", "positive", "text_b")
LABEL_KEYS = ("label", "labels", "gold_label", "relation")


def _try_hf():
    from romteb._hf_datasets import load_dataset

    for src in HF_CANDIDATES:
        try:
            print(f"Trying HF: {src} ...")
            return src, load_dataset(src)
        except Exception as exc:
            print(f"  {src}: {exc}")
    return None, None


def _download_github(url: str) -> bytes:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=300) as resp:
        return resp.read()


def _parse_json_records(blob: bytes, name_hint: str) -> Iterable[dict]:
    text = blob.decode("utf-8", errors="ignore")
    if name_hint.endswith(".jsonl") or "\n{" in text[:500]:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    else:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                yield from data
        except json.JSONDecodeError:
            return


def _from_github() -> dict[str, list[dict]]:
    raw = _download_github(GITHUB_ZIP_URL)
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            lower = name.lower()
            if not (lower.endswith(".json") or lower.endswith(".jsonl")):
                continue
            if "train" in lower:
                key = "train"
            elif "val" in lower or "dev" in lower:
                key = "validation"
            elif "test" in lower:
                key = "test"
            else:
                continue
            with z.open(name) as f:
                blob = f.read()
            for rec in _parse_json_records(blob, lower):
                splits[key].append(rec)
            print(f"  {name}: now {len(splits[key])} {key} records")
    return splits


def _pick(cols: list[str], cands: tuple[str, ...]) -> str:
    for c in cands:
        if c in cols:
            return c
    raise KeyError(f"None of {cands} in {cols}")


def _normalize_records(records: list[dict]) -> Dataset:
    if not records:
        return Dataset.from_list([])
    cols = list(records[0].keys())
    s1k, s2k, lk = _pick(cols, SENT1_KEYS), _pick(cols, SENT2_KEYS), _pick(cols, LABEL_KEYS)
    out = []
    for r in records:
        s1, s2 = (r.get(s1k) or "").strip(), (r.get(s2k) or "").strip()
        if not s1 or not s2:
            continue
        label = LABEL_MAP.get(r.get(lk))
        if label is None and isinstance(r.get(lk), str):
            label = LABEL_MAP.get(r[lk].lower())
        if label is None:
            continue
        out.append({"sentence1": s1, "sentence2": s2, "label": int(label)})
    return Dataset.from_list(out)


def _rebalance_test(test_ds: Dataset, neg_per_pos: int, seed: int) -> Dataset:
    """Undersample contradiction (label=2) so PairClassification is not at prevalence.

    RONLI test has 74 entailment / 96 neutral / 952 contradiction. After the
    PairClassification transform drops neutral, prevalence = 7.2%, so max_ap
    collapses to prevalence for every model. Keep all entailment and neutral
    rows; sub-sample contradiction to ``neg_per_pos * n_entailment`` with a
    fixed seed so revisions are reproducible.
    """
    ent_idx = [i for i, y in enumerate(test_ds["label"]) if y == 0]
    neu_idx = [i for i, y in enumerate(test_ds["label"]) if y == 1]
    con_idx = [i for i, y in enumerate(test_ds["label"]) if y == 2]
    target_con = min(len(con_idx), neg_per_pos * len(ent_idx))
    rng = random.Random(seed)
    con_sample = sorted(rng.sample(con_idx, target_con)) if target_con < len(con_idx) else con_idx
    keep = sorted(set(ent_idx) | set(neu_idx) | set(con_sample))
    balanced = test_ds.select(keep)
    print(
        f"[ronli] test rebalance: entailment={len(ent_idx)} "
        f"contradiction={len(con_idx)}->{len(con_sample)} neutral={len(neu_idx)} "
        f"(pairclassification prevalence after neutral filter: "
        f"{len(ent_idx) / (len(ent_idx) + len(con_sample)):.3f})"
    )
    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--no_rebalance",
        action="store_true",
        help="Push raw upstream test split without undersampling contradiction.",
    )
    args = parser.parse_args()

    src, hf = _try_hf()
    if hf is not None:
        print(f"Using HF source: {src}")
        out = DatasetDict()
        for split in hf:
            recs = list(hf[split])
            out[split] = _normalize_records(recs)
    else:
        print("HF source unavailable; falling back to GitHub.")
        splits = _from_github()
        out = DatasetDict({
            k: _normalize_records(v) for k, v in splits.items() if v
        })

    if "test" not in out:
        if "validation" in out:
            out = DatasetDict(train=out.get("train", out["validation"]), test=out["validation"])
        else:
            base = next(iter(out.values()))
            split = base.train_test_split(test_size=0.1, seed=42)
            out = DatasetDict(train=split["train"], test=split["test"])

    if not args.no_rebalance and "test" in out:
        out["test"] = _rebalance_test(
            out["test"],
            neg_per_pos=REBALANCE_NEG_PER_POS,
            seed=REBALANCE_SEED,
        )

    print_stats("RoNLI", out, text_keys=["sentence1", "sentence2"])

    if args.dry_run:
        return
    commit_msg = (
        "Rebalance RoNLI test (undersample contradiction) for PairClassification"
        if not args.no_rebalance
        else "Refresh RoMTEB RoNLI (3-class NLI)"
    )
    sha = push_to_hub(out, TARGET_REPO, commit_message=commit_msg)
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
