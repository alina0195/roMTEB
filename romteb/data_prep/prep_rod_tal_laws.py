"""Prepare RoD-TAL legal retrieval (text-only) in BEIR format for RoMTEB.

Expected local tree (downloaded from HF):
  datasets/RoD-TAL/
    corpus_laws/corpus_laws.csv
    split_1/queries_split_1_test.csv
    qrels_laws/qrels_queries_split_1_test.csv

Output (single HF repo, three configs):
  - <repo>::corpus   -> {_id, title, text}
  - <repo>::queries  -> {_id, text}
  - <repo>::default  -> {query-id, corpus-id, score}
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from romteb._hf_datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from romteb.data_prep._common import _record_revision


TARGET_REPO = "alina0195/romteb-rod-tal-laws"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = WORKSPACE_ROOT / "datasets" / "RoD-TAL"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def _resolve_query_file(base: Path) -> Path:
    candidates = [
        base / "split_1" / "queries_split_1_test.csv",
        base / "split_1" / "queries_split_1.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No split_1 query file found under {base / 'split_1'}")


def _resolve_qrels_file(base: Path) -> Path:
    candidates = [
        base / "qrels_laws" / "qrels_queries_split_1_test.csv",
        base / "qrels_laws" / "qrels_queries_split_1.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No split_1 qrels file found under {base / 'qrels_laws'}")


def _build_beir(source_dir: Path) -> tuple[Dataset, Dataset, Dataset, dict[str, int]]:
    corpus_rows_raw = _read_csv(source_dir / "corpus_laws" / "corpus_laws.csv")
    query_rows_raw = _read_csv(_resolve_query_file(source_dir))
    qrels_rows_raw = _read_csv(_resolve_qrels_file(source_dir))

    corpus_rows: list[dict[str, str]] = []
    for row in corpus_rows_raw:
        doc_id = (row.get("id") or "").strip()
        text = (row.get("content") or "").strip()
        title = (row.get("title_metadata") or "").strip()
        if not doc_id or not text:
            continue
        corpus_rows.append({"_id": doc_id, "title": title, "text": text})
    corpus_ids = {r["_id"] for r in corpus_rows}

    queries_rows: list[dict[str, str]] = []
    for row in query_rows_raw:
        qid = (row.get("id") or row.get("query_id") or "").strip()
        text = (row.get("question") or row.get("query") or "").strip()
        if not qid or not text:
            continue
        queries_rows.append({"_id": qid, "text": text})
    query_ids = {r["_id"] for r in queries_rows}

    qrels_rows: list[dict[str, str | int]] = []
    skipped_missing = 0
    for row in qrels_rows_raw:
        qid = (row.get("query_id") or row.get("query-id") or "").strip()
        cid = (row.get("corpus_id") or row.get("corpus-id") or "").strip()
        if not qid or not cid:
            skipped_missing += 1
            continue
        if qid not in query_ids or cid not in corpus_ids:
            skipped_missing += 1
            continue
        qrels_rows.append({"query-id": qid, "corpus-id": cid, "score": 1})

    stats = {
        "corpus": len(corpus_rows),
        "queries": len(queries_rows),
        "qrels": len(qrels_rows),
        "skipped_qrels": skipped_missing,
    }
    return (
        Dataset.from_list(corpus_rows),
        Dataset.from_list(queries_rows),
        Dataset.from_list(qrels_rows),
        stats,
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
    parser.add_argument(
        "--source_dir",
        default=str(DEFAULT_SOURCE),
        help="Path to local RoD-TAL directory",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise SystemExit(f"Source dir not found: {source_dir}")

    corpus, queries, qrels, stats = _build_beir(source_dir)
    print(
        "BEIR shape: "
        f"corpus={stats['corpus']:,}  queries={stats['queries']:,}  "
        f"qrels={stats['qrels']:,}  skipped_qrels={stats['skipped_qrels']:,}"
    )

    if args.dry_run:
        return
    sha = _push_multi_config(corpus, queries, qrels, TARGET_REPO)
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
