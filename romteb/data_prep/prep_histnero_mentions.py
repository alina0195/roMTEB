"""HistNERo mention-typing: gold NER spans as classification examples.

Not token-level NER. Each annotated mention becomes one row:

    text  = "Mențiune: {span}\\nContext: {sentence}"
    label = PERS | ORG | LOC | PROD | DATE

A frozen embedder + logistic regression (MTEB Classification) predicts
the entity type of the gold span in context.

Source: avramandrei/histnero
Output: alina0195/romteb-histnero-mentions
"""

from __future__ import annotations

import argparse
from collections import Counter

from romteb._hf_datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from romteb.data_prep._common import print_stats, push_to_hub


SOURCE_REPO = "avramandrei/histnero"
TARGET_REPO = "alina0195/romteb-histnero-mentions"

# HistNERo ClassLabel names (see dataset card).
TAG_TO_TYPE = {
    "B-PERS": "PERS",
    "I-PERS": "PERS",
    "B-ORG": "ORG",
    "I-ORG": "ORG",
    "B-LOC": "LOC",
    "I-LOC": "LOC",
    "B-PROD": "PROD",
    "I-PROD": "PROD",
    "B-DATE": "DATE",
    "I-DATE": "DATE",
}

TYPE_TO_ID = {"PERS": 0, "ORG": 1, "LOC": 2, "PROD": 3, "DATE": 4}
ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}


def _tag_name(tag, names: list[str] | None) -> str:
    if isinstance(tag, str):
        return tag
    if names is not None and isinstance(tag, int) and 0 <= tag < len(names):
        return names[tag]
    return str(tag)


def _mentions_from_bio(tokens: list[str], tags: list, names: list[str] | None) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    current_type: str | None = None
    current_toks: list[str] = []

    def flush():
        nonlocal current_type, current_toks
        if current_type and current_toks:
            span = " ".join(current_toks).strip()
            if span:
                mentions.append((span, current_type))
        current_type = None
        current_toks = []

    for tok, tag in zip(tokens, tags):
        name = _tag_name(tag, names)
        etype = TAG_TO_TYPE.get(name)
        is_begin = isinstance(name, str) and name.startswith("B-")
        if etype is None:
            flush()
            continue
        if is_begin or current_type != etype:
            flush()
            current_type = etype
            current_toks = [tok]
        else:
            current_toks.append(tok)
    flush()
    return mentions


def _split_to_examples(ds, names: list[str] | None) -> Dataset:
    rows = []
    for item in ds:
        tokens = list(item["tokens"])
        tags = list(item["ner_tags"])
        sentence = " ".join(tokens)
        for span, etype in _mentions_from_bio(tokens, tags, names):
            rows.append(
                {
                    "text": f"Mențiune: {span}\nContext: {sentence}",
                    "label": TYPE_TO_ID[etype],
                    "label_name": etype,
                    "span": span,
                    "region": item.get("region") or "",
                }
            )
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Loading {SOURCE_REPO} ...")
    raw = load_dataset(SOURCE_REPO)
    names = None
    try:
        names = raw["train"].features["ner_tags"].feature.names
        print(f"  tag names: {names}")
    except Exception:
        print("  (could not read ClassLabel names; using integer fallback)")

    train = _split_to_examples(raw["train"], names)
    if "valid" in raw:
        valid = _split_to_examples(raw["valid"], names)
        # Give the probe more train signal; MTEB only uses `train`.
        train = concatenate_datasets([train, valid])
    test = _split_to_examples(raw["test"], names)
    out = DatasetDict(train=train, test=test)

    for split, ds in out.items():
        print(f"  {split}: {len(ds):,} mentions  {dict(Counter(ds['label_name']))}")
    print_stats("HistNERo mention-typing", out, text_keys=["text"])

    if args.dry_run:
        return
    sha = push_to_hub(
        out,
        TARGET_REPO,
        commit_message="RoMTEB HistNERo mention classification (span + context)",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
