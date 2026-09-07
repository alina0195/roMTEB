"""Prepare RORetrieval articles as two clustering tasks (outlet + type).

The local corpus path is a TODO until it is wired in; pass ``--input``.

Expected records (JSON / JSONL / parquet / directory of those):
  text / content / body / article  — article text (title is prepended if present)
  outlet / publication / source / newspaper — e.g. adevarul, digi24
  type / category / genre — e.g. news, recipes, stories

USAGE:
    ./apptainer-exec-romteb.sh romteb/data_prep/prep_roretrieval_clustering.py \\
        --input /path/to/RORetrieval --dry_run
    ./apptainer-exec-romteb.sh romteb/data_prep/prep_roretrieval_clustering.py \\
        --input /path/to/RORetrieval
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from romteb._hf_datasets import Dataset, DatasetDict
from romteb.data_prep._common import print_stats, push_to_hub

OUTLET_REPO = "alina0195/romteb-roretrieval-outlet-clustering"
TYPE_REPO = "alina0195/romteb-roretrieval-type-clustering"

TEXT_KEYS = ("text", "content", "body", "article", "document")
TITLE_KEYS = ("title", "headline")
OUTLET_KEYS = ("outlet", "publication", "source", "newspaper", "publisher", "site")
TYPE_KEYS = ("type", "category", "genre", "content_type", "doc_type")

KNOWN_OUTLETS = (
    "adevarul",
    "digi24",
    "protv",
    "mediafax",
    "libertatea",
    "zf",
    "evz",
    "cotidianul",
    "hotnews",
    "gsp",
    "stirileprotv",
)
KNOWN_TYPES = ("news", "recipes", "stories", "recipe", "story")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    folded = value.strip().lower()
    return _SLUG_RE.sub("", folded)


def _pick(record: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return str(record[key]).strip()
        lower = {k.lower(): k for k in record}
        if key in lower and record[lower[key]] not in (None, ""):
            return str(record[lower[key]]).strip()
    return None


def _iter_json_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        yield from raw
    elif isinstance(raw, dict):
        for key in ("data", "articles", "documents", "items"):
            if isinstance(raw.get(key), list):
                yield from raw[key]
                return
        yield raw


def _collect_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    for pattern in ("*.jsonl", "*.json", "*.parquet"):
        files.extend(root.rglob(pattern))
    return sorted(files)


def _load_records(input_path: Path) -> list[dict]:
    files = _collect_files(input_path)
    if not files:
        raise SystemExit(f"No json/jsonl/parquet files under {input_path}")
    rows: list[dict] = []
    for path in files:
        if path.suffix.lower() == ".parquet":
            try:
                import pandas as pd

                frame = pd.read_parquet(path)
                rows.extend(frame.to_dict(orient="records"))
            except Exception as exc:
                print(f"  skip {path}: {exc}")
            continue
        try:
            rows.extend(_iter_json_records(path))
        except Exception as exc:
            print(f"  skip {path}: {exc}")
    return rows


def _normalize_outlet(raw: str) -> str | None:
    slug = _slug(raw)
    if not slug:
        return None
    for known in KNOWN_OUTLETS:
        if known in slug or slug in known:
            return known
    return slug


def _normalize_type(raw: str) -> str | None:
    slug = _slug(raw)
    mapping = {"recipe": "recipes", "story": "stories", "stire": "news"}
    slug = mapping.get(slug, slug)
    if slug in {"news", "recipes", "stories"}:
        return slug
    if slug:
        return slug
    return None


def _to_examples(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        text = _pick(rec, TEXT_KEYS)
        title = _pick(rec, TITLE_KEYS) or ""
        if not text:
            continue
        sentence = f"{title}\n{text}".strip() if title and title not in text else text
        outlet = _pick(rec, OUTLET_KEYS)
        typ = _pick(rec, TYPE_KEYS)
        out.append(
            {
                "sentences": sentence,
                "outlet": _normalize_outlet(outlet) if outlet else None,
                "content_type": _normalize_type(typ) if typ else None,
            }
        )
    return out


def _balanced_subsample(
    examples: list[dict],
    label_key: str,
    max_per_label: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        label = ex.get(label_key)
        if not label:
            continue
        by_label[str(label)].append(ex)
    sampled = []
    for label, items in by_label.items():
        rng.shuffle(items)
        sampled.extend(items[:max_per_label])
    rng.shuffle(sampled)
    return sampled


def _clustering_split(examples: list[dict], label_key: str) -> DatasetDict:
    rows = [
        {"sentences": ex["sentences"], "labels": ex[label_key]}
        for ex in examples
        if ex.get(label_key)
    ]
    if len(rows) < 20:
        raise SystemExit(
            f"Not enough labelled rows for {label_key} ({len(rows)}). "
            "Check --input schema."
        )
    return DatasetDict({"test": Dataset.from_list(rows)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="TODO: local path to the RORetrieval corpus (file or directory).",
    )
    parser.add_argument("--max_per_label", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        raise SystemExit(
            f"RORetrieval path not found: {root}. Pass --input when the corpus "
            "is available."
        )

    print(f"Loading records from {root} ...")
    examples = _to_examples(_load_records(root))
    print(f"  parsed {len(examples):,} articles")
    print("  outlets:", Counter(ex["outlet"] for ex in examples if ex["outlet"]))
    print("  types:", Counter(ex["content_type"] for ex in examples if ex["content_type"]))

    outlet_ex = _balanced_subsample(examples, "outlet", args.max_per_label, args.seed)
    type_ex = _balanced_subsample(examples, "content_type", args.max_per_label, args.seed)
    outlet_ds = _clustering_split(outlet_ex, "outlet")
    type_ds = _clustering_split(type_ex, "content_type")
    print_stats("RoNewsOutletClusteringP2P", outlet_ds, text_keys=["sentences"])
    print_stats("RoNewsTypeClusteringP2P", type_ds, text_keys=["sentences"])

    if args.dry_run:
        print("dry_run: not pushing")
        return
    push_to_hub(outlet_ds, OUTLET_REPO, commit_message="RORetrieval outlet clustering")
    push_to_hub(type_ds, TYPE_REPO, commit_message="RORetrieval type clustering")


if __name__ == "__main__":
    main()
