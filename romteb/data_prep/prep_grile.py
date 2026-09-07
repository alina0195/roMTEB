"""Prepare GRILE grammar MCQs as BEIR reranking + retrieval.

Source: datasets/GRILE/grammar_questions_explained.json
  query  = grammar question
  corpus = one document per (question, option)
  qrels  = the correct option
  top_ranked = the 4 options of that question (reranking only)

Reranking is the natural exam protocol. Retrieval searches the full option bank.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from romteb._hf_datasets import Dataset

from romteb.data_prep._beir_io import print_beir_stats, push_beir_repo


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = WORKSPACE_ROOT / "datasets" / "GRILE" / "grammar_questions_explained.json"
RERANKING_REPO = "alina0195/romteb-grile-reranking"
RETRIEVAL_REPO = "alina0195/romteb-grile-retrieval"


def _parse_options(raw) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k).lower(): str(v).strip() for k, v in raw.items() if str(v).strip()}
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return {str(k).lower(): str(v).strip() for k, v in parsed.items() if str(v).strip()}
    return {}


def _build_beir(records: list[dict]):
    corpus_rows: list[dict] = []
    queries_rows: list[dict] = []
    qrels_rows: list[dict] = []
    top_ranked_rows: list[dict] = []
    skipped = 0

    for i, item in enumerate(records):
        question = (item.get("Question") or "").strip()
        options = _parse_options(item.get("Options"))
        gold = str(item.get("Correct_answer") or "").strip().lower()
        if not question or len(options) < 2 or gold not in options:
            skipped += 1
            continue

        qid = f"q_{item.get('ID_Entry') or i}"
        queries_rows.append({"_id": qid, "text": question})
        cand_ids: list[str] = []
        for letter, text in sorted(options.items()):
            doc_id = f"{qid}_{letter}"
            corpus_rows.append({"_id": doc_id, "title": "", "text": text})
            cand_ids.append(doc_id)
        qrels_rows.append({"query-id": qid, "corpus-id": f"{qid}_{gold}", "score": 1})
        top_ranked_rows.append({"query-id": qid, "corpus-ids": cand_ids})

    stats = {"kept": len(queries_rows), "skipped": skipped}
    return (
        Dataset.from_list(corpus_rows),
        Dataset.from_list(queries_rows),
        Dataset.from_list(qrels_rows),
        Dataset.from_list(top_ranked_rows),
        stats,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_reranking", action="store_true")
    parser.add_argument("--skip_retrieval", action="store_true")
    args = parser.parse_args()

    path = Path(args.source)
    if not path.exists():
        raise FileNotFoundError(f"GRILE source not found: {path}")
    print(f"Loading {path} ...")
    records = json.loads(path.read_text(encoding="utf-8"))
    print(f"  {len(records):,} raw questions")

    corpus, queries, qrels, top_ranked, stats = _build_beir(records)
    print_beir_stats("GRILE", corpus, queries, qrels, top_ranked, stats)

    if args.dry_run:
        return
    if not args.skip_reranking:
        sha = push_beir_repo(
            corpus, queries, qrels, RERANKING_REPO, top_ranked=top_ranked
        )
        print(f"\nReranking pin: revision = {sha!r}")
    if not args.skip_retrieval:
        sha_r = push_beir_repo(
            corpus, queries, qrels, RETRIEVAL_REPO, top_ranked=None
        )
        print(f"\nRetrieval pin: revision = {sha_r!r}")


if __name__ == "__main__":
    main()
