"""Build MS MARCO RO retrieval from the official translated test split.

Source: ``alina0195/ro-msmarco-divided`` (anchor / positive / negative).
Splits: train ~11.6M, eval ~34.8k, test ~36.3k triplets.

Eval protocol:
  - queries = unique ``anchor`` texts from **test**
  - qrels   = that query's unique ``positive`` passage(s)
  - corpus  = unique positives + negatives from test
              (plus unique eval passages as extra distractors, default on)
  - no ``top_ranked`` (full retrieval, nDCG@10)

Train is not indexed (11.6M triplets). The old hash-holdout from
``alina0195/ro-msmarco`` train lives in ``prep_msmarco_ro_archived.py``.

Output BEIR repo: alina0195/romteb-msmarco-ro-retrieval
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict

import pandas as pd


SOURCE_REPO = "alina0195/ro-msmarco-divided"
QUERY_SPLIT = "test"
TARGET_REPO = "alina0195/romteb-msmarco-ro-retrieval"

# Load converted parquet shards so we never pull the 11.6M-row train split.
_PARQUET = {
    "test": (
        "https://huggingface.co/datasets/alina0195/ro-msmarco-divided/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
    ),
    "eval": (
        "https://huggingface.co/datasets/alina0195/ro-msmarco-divided/resolve/"
        "refs%2Fconvert%2Fparquet/default/eval/0000.parquet"
    ),
}


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _norm(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _add_passage(corpus_by_id: dict[str, str], text: str) -> str | None:
    if not text:
        return None
    pid = _stable_id("p", text)
    corpus_by_id.setdefault(pid, text)
    return pid


def _print_stats(corpus_rows, queries_rows, qrels_rows, extra: dict) -> None:
    print("\n=== MS MARCO RO (divided test) ===")
    print(
        f"  corpus={len(corpus_rows):,}  queries={len(queries_rows):,}  "
        f"qrels={len(qrels_rows):,}  (no top_ranked)"
    )
    for k, v in extra.items():
        print(f"  {k}: {v}")


def _load_split(split: str) -> pd.DataFrame:
    url = _PARQUET[split]
    print(
        f"Loading {SOURCE_REPO} split={split} parquet ({url.split('/')[-1]}) ...",
        flush=True,
    )
    return pd.read_parquet(url)


def _collect(
    max_test_queries: int | None,
    include_eval_corpus: bool,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    test = _load_split(QUERY_SPLIT)

    # qid -> {text, pos_ids}
    bucket: dict[str, dict] = {}
    corpus_by_id: dict[str, str] = {}
    n_test_rows = 0
    n_skipped = 0
    for anchor, positive, negative in zip(
        test["anchor"].tolist(),
        test["positive"].tolist(),
        test["negative"].tolist(),
    ):
        n_test_rows += 1
        anchor = _norm(anchor)
        positive = _norm(positive)
        negative = _norm(negative)
        if not anchor or not positive:
            n_skipped += 1
            continue
        qid = _stable_id("q", anchor)
        rec = bucket.get(qid)
        if rec is None:
            rec = {"text": anchor, "pos_ids": set()}
            bucket[qid] = rec
        pid = _add_passage(corpus_by_id, positive)
        if pid:
            rec["pos_ids"].add(pid)
        if negative and negative != positive:
            _add_passage(corpus_by_id, negative)

    eval_anchors: set[str] = set()
    n_eval_passages_before = len(corpus_by_id)
    n_eval_rows = 0
    if include_eval_corpus:
        eval_ds = _load_split("eval")
        for anchor, positive, negative in zip(
            eval_ds["anchor"].tolist(),
            eval_ds["positive"].tolist(),
            eval_ds["negative"].tolist(),
        ):
            n_eval_rows += 1
            anchor = _norm(anchor)
            if anchor:
                eval_anchors.add(_stable_id("q", anchor))
            positive = _norm(positive)
            _add_passage(corpus_by_id, positive)
            negative = _norm(negative)
            if negative and negative != positive:
                _add_passage(corpus_by_id, negative)

    qids = sorted(bucket)
    if max_test_queries is not None:
        qids = qids[: max(0, max_test_queries)]

    queries_rows = []
    qrels_rows = []
    n_multi_pos = 0
    n_qrels_by_q: dict[str, int] = defaultdict(int)
    for qid in qids:
        rec = bucket[qid]
        queries_rows.append({"_id": qid, "text": rec["text"]})
        if len(rec["pos_ids"]) > 1:
            n_multi_pos += 1
        for pid in sorted(rec["pos_ids"]):
            qrels_rows.append({"query-id": qid, "corpus-id": pid, "score": 1})
            n_qrels_by_q[qid] += 1

    overlap_eval = sum(1 for qid in qids if qid in eval_anchors)

    corpus_rows = [
        {"_id": did, "title": "", "text": text} for did, text in corpus_by_id.items()
    ]
    stats = {
        "source": SOURCE_REPO,
        "query_split": QUERY_SPLIT,
        "test_rows": n_test_rows,
        "skipped_empty": n_skipped,
        "unique_test_queries": len(bucket),
        "queries_kept": len(queries_rows),
        "queries_with_multiple_golds": n_multi_pos,
        "eval_rows": n_eval_rows,
        "eval_passages_added": max(0, len(corpus_by_id) - n_eval_passages_before),
        "test_query_overlap_eval": overlap_eval,
        "mean_golds_per_query": (
            sum(n_qrels_by_q.values()) / max(len(n_qrels_by_q), 1)
        ),
    }
    return corpus_rows, queries_rows, qrels_rows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--max_test_queries",
        type=int,
        default=None,
        help="Optional cap on unique test queries (stable id sort). Default: all.",
    )
    parser.add_argument(
        "--no_eval_corpus",
        action="store_true",
        help="Index only test passages (no extra eval-split distractors).",
    )
    args = parser.parse_args()
    print("Building MS MARCO RO retrieval from ro-msmarco-divided test ...", flush=True)

    corpus_rows, queries_rows, qrels_rows, stats = _collect(
        max_test_queries=args.max_test_queries,
        include_eval_corpus=not args.no_eval_corpus,
    )
    _print_stats(
        corpus_rows,
        queries_rows,
        qrels_rows,
        stats,
    )
    if len(queries_rows) == 0:
        raise SystemExit("No test queries in alina0195/ro-msmarco-divided split=test.")
    if int(stats["test_query_overlap_eval"]) > 0:
        print(
            f"WARNING: {stats['test_query_overlap_eval']} test queries also "
            "appear as eval anchors (query-level leakage in the source split)."
        )

    if args.dry_run:
        q0 = queries_rows[0]
        g0 = qrels_rows[0]
        print(f'  example query="{q0["text"][:90]}..."')
        print(f'  gold corpus-id={g0["corpus-id"]} (query-id={g0["query-id"]})')
        return

    from romteb._hf_datasets import Dataset
    from romteb.data_prep._beir_io import push_beir_repo

    corpus = Dataset.from_list(corpus_rows)
    queries = Dataset.from_list(queries_rows)
    qrels = Dataset.from_list(qrels_rows)
    sha = push_beir_repo(
        corpus,
        queries,
        qrels,
        TARGET_REPO,
        top_ranked=None,
        commit_message="MS MARCO RO retrieval from ro-msmarco-divided test",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
