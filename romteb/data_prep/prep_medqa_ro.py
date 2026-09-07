"""Prepare MedQARo retrieval in BEIR format for RoMTEB.

Source files (local workspace):
  - /datasets/MedQARo Dataset/romedqa_train_dataset.csv
  - /datasets/MedQARo Dataset/romedqa_val_dataset.csv
  - /datasets/MedQARo Dataset/romedqa_test_dataset.csv

Output (single HF repo, three configs):
  - alina0195/romteb-medqa-ro :: corpus  -> {_id, title, text}
  - alina0195/romteb-medqa-ro :: queries -> {_id, text}
  - alina0195/romteb-medqa-ro :: default -> {query-id, corpus-id, score}
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from romteb._hf_datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from romteb.data_prep._common import _record_revision


TARGET_REPO = "alina0195/romteb-medqa-ro"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = WORKSPACE_ROOT / "datasets" / "MedQARo Dataset"
SOURCE_SPLITS = {
    "train": SOURCE_DIR / "romedqa_train_dataset.csv",
    "validation": SOURCE_DIR / "romedqa_val_dataset.csv",
    "test": SOURCE_DIR / "romedqa_test_dataset.csv",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            context = (row.get("Epicriza") or "").strip()
            question = (row.get("Intrebare") or "").strip()
            answer = (row.get("Raspuns") or "").strip()
            if not context or not question:
                continue
            rows.append({"context": context, "question": question, "answer": answer})
    return rows


def _build_beir(test_rows: list[dict[str, str]], corpus_rows: list[dict[str, str]]) -> tuple[Dataset, Dataset, Dataset]:
    corpus_map: dict[str, str] = {}
    corpus_data: list[dict[str, str]] = []
    for row in corpus_rows:
        text = row["context"]
        if text not in corpus_map:
            doc_id = f"doc_{len(corpus_map)}"
            corpus_map[text] = doc_id
            corpus_data.append({"_id": doc_id, "title": "", "text": text})

    queries_data: list[dict[str, str]] = []
    qrels_data: list[dict[str, str | int]] = []
    for i, row in enumerate(test_rows):
        qid = f"q_{i}"
        doc_id = corpus_map[row["context"]]
        queries_data.append({"_id": qid, "text": row["question"]})
        qrels_data.append({"query-id": qid, "corpus-id": doc_id, "score": 1})

    return (
        Dataset.from_list(corpus_data),
        Dataset.from_list(queries_data),
        Dataset.from_list(qrels_data),
    )


def _push_multi_config(corpus: Dataset, queries: Dataset, qrels: Dataset, repo_id: str) -> str:
    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    print(f"Pushing corpus  ({len(corpus):,} docs)")
    corpus.push_to_hub(repo_id, config_name="corpus", split="corpus")

    print(f"Pushing queries ({len(queries):,} queries)")
    queries.push_to_hub(repo_id, config_name="queries", split="queries")

    print(f"Pushing qrels   ({len(qrels):,} judgements)")
    DatasetDict({"test": qrels}).push_to_hub(repo_id, config_name="default")

    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    sha = refs.branches[0].target_commit if refs.branches else "main"
    _record_revision(repo_id, sha)
    print(f"Pushed: https://huggingface.co/datasets/{repo_id}  (rev {sha})")
    return sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--test_only_corpus",
        action="store_true",
        help="Build the retrieval corpus only from the test split contexts.",
    )
    args = parser.parse_args()

    loaded = {split: _load_rows(path) for split, path in SOURCE_SPLITS.items()}
    print(
        "Loaded MedQARo rows: "
        + ", ".join(f"{split}={len(rows):,}" for split, rows in loaded.items())
    )

    test_rows = loaded["test"]
    corpus_source = test_rows if args.test_only_corpus else (
        loaded["train"] + loaded["validation"] + loaded["test"]
    )

    corpus, queries, qrels = _build_beir(test_rows=test_rows, corpus_rows=corpus_source)
    print(
        f"BEIR shape: corpus={len(corpus):,} docs  queries={len(queries):,}  qrels={len(qrels):,}"
    )

    if args.dry_run:
        return

    sha = _push_multi_config(corpus, queries, qrels, TARGET_REPO)
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
