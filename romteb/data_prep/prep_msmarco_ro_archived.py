"""ARCHIVED — hash-holdout from alina0195/ro-msmarco train triplets.

Replaced by prep_msmarco_ro.py, which uses the official test split of
alina0195/ro-msmarco-divided.
"""

from __future__ import annotations

import argparse
import hashlib
from romteb._hf_datasets import Dataset, load_dataset

from romteb.data_prep._beir_io import print_beir_stats, push_beir_repo


SOURCE_REPO = "alina0195/ro-msmarco"
TARGET_REPO = "alina0195/romteb-msmarco-ro-retrieval"


def _is_test_query(text: str, modulo: int = 20) -> bool:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo == 0


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _collect(
    max_test_queries: int,
    negs_per_query: int,
    max_rows: int | None,
    scan_modulo: int,
) -> tuple[Dataset, Dataset, Dataset, dict[str, int]]:
    print(f"Streaming {SOURCE_REPO} split=train ...")
    ds = load_dataset(SOURCE_REPO, split="train", streaming=True)

    bucket: dict[str, dict] = {}
    n_rows = 0
    n_unique_seen = 0

    for row in ds:
        n_rows += 1
        if max_rows is not None and n_rows > max_rows:
            break
        anchor = (row.get("anchor") or "").strip()
        positive = (row.get("positive") or "").strip()
        negative = (row.get("negative") or "").strip()
        if not anchor or not positive:
            continue
        if not _is_test_query(anchor, scan_modulo):
            continue

        qid = _stable_id("q", anchor)
        if qid not in bucket:
            if len(bucket) >= max_test_queries:
                if all(len(v["negs"]) >= negs_per_query for v in bucket.values()):
                    break
                continue
            bucket[qid] = {"text": anchor, "pos": positive, "negs": set()}
            n_unique_seen += 1
        if negative and negative != positive:
            if len(bucket[qid]["negs"]) < negs_per_query:
                bucket[qid]["negs"].add(negative)

        if n_rows % 200_000 == 0:
            print(
                f"  scanned {n_rows:,} rows; test queries={len(bucket):,} "
                f"(target {max_test_queries})"
            )

    queries_rows = []
    corpus_by_id: dict[str, str] = {}
    qrels_rows = []
    for qid, rec in bucket.items():
        queries_rows.append({"_id": qid, "text": rec["text"]})
        pid = _stable_id("p", rec["pos"])
        corpus_by_id[pid] = rec["pos"]
        qrels_rows.append({"query-id": qid, "corpus-id": pid, "score": 1})
        for neg in rec["negs"]:
            nid = _stable_id("p", neg)
            corpus_by_id.setdefault(nid, neg)

    corpus = Dataset.from_list(
        [{"_id": did, "title": "", "text": text} for did, text in corpus_by_id.items()]
    )
    queries = Dataset.from_list(queries_rows)
    qrels = Dataset.from_list(qrels_rows)
    stats = {
        "rows_scanned": n_rows,
        "test_queries": len(queries_rows),
        "mean_negs": (
            sum(len(v["negs"]) for v in bucket.values()) / max(len(bucket), 1)
        ),
    }
    return corpus, queries, qrels, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_test_queries", type=int, default=3000)
    parser.add_argument("--negs_per_query", type=int, default=16)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--scan_modulo", type=int, default=20)
    args = parser.parse_args()

    corpus, queries, qrels, stats = _collect(
        max_test_queries=args.max_test_queries,
        negs_per_query=args.negs_per_query,
        max_rows=args.max_rows,
        scan_modulo=args.scan_modulo,
    )
    print_beir_stats("MS MARCO RO (archived hash-holdout)", corpus, queries, qrels, extra=stats)
    if len(queries) == 0:
        raise SystemExit("No test queries collected; increase --max_rows.")
    if args.dry_run:
        return
    sha = push_beir_repo(corpus, queries, qrels, TARGET_REPO, top_ranked=None)
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
