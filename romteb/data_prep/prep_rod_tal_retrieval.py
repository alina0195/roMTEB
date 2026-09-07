"""Prepare RoD-TAL legal IR as a BEIR-format retrieval dataset for RoMTEB.

Source: ``GRAI-UNSTPB/RoD-TAL`` (gated, CC-BY-NC-SA-4.0).

Uses the text-only legal retrieval components:
  - ``corpus_laws`` config  -> legal article passages (the corpus)
  - ``split_1``   config    -> driving-exam questions (queries)
  - ``qrels_laws`` config   -> relevance judgements (query_id -> corpus_id, score=1)

The qrels contain only positive matches (score=1). MTEB infers negatives
automatically: any corpus doc absent from a query's qrels is scored 0.

Output (single HF repo, three configs — standard BEIR layout):
  - alina0195/romteb-rod-tal-retrieval :: corpus  -> {_id, title, text}
  - alina0195/romteb-rod-tal-retrieval :: queries -> {_id, text}
  - alina0195/romteb-rod-tal-retrieval :: default -> {query-id, corpus-id, score}
"""

from __future__ import annotations

import argparse

from romteb._hf_datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

from romteb.data_prep._common import _record_revision

SOURCE_REPO = "GRAI-UNSTPB/RoD-TAL"
TARGET_REPO = "alina0195/romteb-rod-tal-retrieval"


def _load_corpus():
    """Load corpus_laws and normalise to {_id, title, text}."""
    ds = load_dataset(SOURCE_REPO, "corpus_laws", split="corpus")
    cols = ds.column_names
    print(f"corpus_laws columns: {cols}, rows: {len(ds):,}")

    id_col = next((c for c in ("_id", "id", "doc_id", "corpus_id") if c in cols), None)
    text_col = next((c for c in ("text", "content", "article_text", "passage") if c in cols), None)
    title_col = next((c for c in ("title", "article_title") if c in cols), None)

    if id_col is None:
        print(f"  No ID column found; generating sequential IDs (doc_0, doc_1, ...)")
    if text_col is None:
        raise SystemExit(f"Cannot find text column in corpus_laws: {cols}")

    rows = []
    for i, item in enumerate(ds):
        doc_id = str(item[id_col]) if id_col else f"doc_{i}"
        text = (item[text_col] or "").strip()
        title = (item[title_col] or "").strip() if title_col else ""
        if not text:
            continue
        rows.append({"_id": doc_id, "title": title, "text": text})

    print(f"  Corpus: {len(rows):,} documents after filtering empty")
    return Dataset.from_list(rows)


def _load_queries(split_name: str):
    """Load query split and normalise to {_id, text}."""
    ds = load_dataset(SOURCE_REPO, "split_1", split=split_name)
    cols = ds.column_names
    print(f"split_1/{split_name} columns: {cols}, rows: {len(ds):,}")

    id_col = next((c for c in ("_id", "id", "query_id", "question_id") if c in cols), None)
    text_col = next((c for c in ("question", "text", "query") if c in cols), None)

    if id_col is None:
        print(f"  No ID column found; generating sequential IDs (q_0, q_1, ...)")
    if text_col is None:
        raise SystemExit(f"Cannot find text column in split_1/{split_name}: {cols}")

    rows = []
    for i, item in enumerate(ds):
        qid = str(item[id_col]) if id_col else f"q_{i}"
        text = (item[text_col] or "").strip()
        if not text:
            continue
        rows.append({"_id": qid, "text": text})

    print(f"  Queries ({split_name}): {len(rows):,} after filtering empty")
    return Dataset.from_list(rows)


def _load_qrels(split_name: str):
    """Load qrels and normalise to {query-id, corpus-id, score}."""
    ds = load_dataset(SOURCE_REPO, "qrels_laws", split=split_name)
    cols = ds.column_names
    print(f"qrels_laws/{split_name} columns: {cols}, rows: {len(ds):,}")

    qid_col = next((c for c in ("query_id", "query-id", "qid") if c in cols), None)
    cid_col = next((c for c in ("corpus_id", "corpus-id", "doc_id", "docid") if c in cols), None)
    score_col = next((c for c in ("score", "relevance", "label") if c in cols), None)

    if qid_col is None or cid_col is None:
        raise SystemExit(f"Cannot find query_id/corpus_id columns in qrels_laws/{split_name}: {cols}")

    rows = []
    for item in ds:
        qid = str(item[qid_col])
        cid = str(item[cid_col])
        score = int(item[score_col]) if score_col else 1
        rows.append({"query-id": qid, "corpus-id": cid, "score": score})

    unique_queries = len({r["query-id"] for r in rows})
    unique_docs = len({r["corpus-id"] for r in rows})
    print(f"  Qrels ({split_name}): {len(rows):,} judgements, "
          f"{unique_queries:,} unique queries, {unique_docs:,} unique docs")
    return Dataset.from_list(rows)


def _push_multi_config(
    corpus: Dataset,
    queries: Dataset,
    qrels_test: Dataset,
    repo_id: str,
) -> str:
    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    print(f"Pushing corpus  ({len(corpus):,} docs)")
    corpus.push_to_hub(repo_id, config_name="corpus", split="corpus")

    print(f"Pushing queries ({len(queries):,} queries)")
    queries.push_to_hub(repo_id, config_name="queries", split="queries")

    print(f"Pushing qrels   ({len(qrels_test):,} judgements)")
    DatasetDict({"test": qrels_test}).push_to_hub(repo_id, config_name="default")

    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    sha = refs.branches[0].target_commit if refs.branches else "main"
    _record_revision(repo_id, sha)
    print(f"Pushed: https://huggingface.co/datasets/{repo_id}  (rev {sha})")
    return sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Loading {SOURCE_REPO} (gated — access required) ...")

    corpus = _load_corpus()
    queries_test = _load_queries("split_1_test")
    qrels_test = _load_qrels("qrels_queries_split_1_test")

    # Validate: all qrel query IDs should appear in queries
    query_ids = {r["_id"] for r in queries_test}
    qrel_query_ids = {r["query-id"] for r in qrels_test}
    orphan_qids = qrel_query_ids - query_ids
    if orphan_qids:
        print(f"  WARNING: {len(orphan_qids)} qrel query_ids not found in queries")

    corpus_ids = {r["_id"] for r in corpus}
    qrel_corpus_ids = {r["corpus-id"] for r in qrels_test}
    orphan_cids = qrel_corpus_ids - corpus_ids
    if orphan_cids:
        print(f"  WARNING: {len(orphan_cids)} qrel corpus_ids not found in corpus")

    print(f"\nBEIR shape: corpus={len(corpus):,} docs  "
          f"queries={len(queries_test):,}  qrels={len(qrels_test):,}")

    if args.dry_run:
        return

    sha = _push_multi_config(corpus, queries_test, qrels_test, TARGET_REPO)
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
