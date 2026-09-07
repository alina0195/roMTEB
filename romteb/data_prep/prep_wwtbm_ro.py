"""Prepare WWTBM-Romanian multiple-choice QA in BEIR-style reranking format.

Source: ``WWTBM/wwtbm`` config ``Romanian`` (~1000 four-choice questions
from the Romanian edition of "Who Wants to Be a Millionaire?",
annotated with category / difficulty / cultural_context).
See https://arxiv.org/abs/2506.05991 for the paper.

This is a *cultural* MCQA task, not open-domain retrieval. Each question
has exactly four candidate answers; the embedding model must rank the
gold answer above the three distractors for that *specific* question.
We encode that constraint with a ``top_ranked`` config (per-query
candidate pool of 4), so MTEB never compares an answer to candidates
from a different question. Equivalent metric: accuracy = recall@1 with
4 candidates.

Output (single HF repo, four configs):
  - alina0195/romteb-wwtbm-ro :: corpus     -> {_id, title, text}
                                               (1 row per (question, option) pair)
  - alina0195/romteb-wwtbm-ro :: queries    -> {_id, text}
                                               (1 row per question)
  - alina0195/romteb-wwtbm-ro :: default    -> {query-id, corpus-id, score}
                                               (qrels; score=1 for the gold option)
  - alina0195/romteb-wwtbm-ro :: top_ranked -> {query-id, corpus-ids}
                                               (the 4 options of *that* question)

Per-question candidate IDs (``q{i}_a{j}``) are required because the
same answer surface form (e.g. "Italia") may appear as a distractor in
one question and as the gold answer in another; a global corpus dedupe
would conflate them.
"""

from __future__ import annotations

import argparse
from collections import Counter

from romteb._hf_datasets import Dataset, load_dataset

from romteb.data_prep._beir_io import print_beir_stats, push_beir_repo


SOURCE_REPO = "WWTBM/wwtbm"
SOURCE_CONFIG = "Romanian"
SOURCE_SPLIT = "train"
TARGET_REPO = "alina0195/romteb-wwtbm-ro"
RETRIEVAL_REPO = "alina0195/romteb-wwtbm-ro-retrieval"


def _build_beir(
    ds,
) -> tuple[Dataset, Dataset, Dataset, Dataset, dict[str, int]]:
    """Build corpus / queries / qrels / top_ranked datasets.

    Returns the four datasets plus a small stats dict for the summary.
    """
    corpus_rows: list[dict] = []
    queries_rows: list[dict] = []
    qrels_rows: list[dict] = []
    top_ranked_rows: list[dict] = []

    skipped_no_match = 0
    skipped_wrong_arity = 0
    culture_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()

    for i, item in enumerate(ds):
        question = (item.get("question") or "").strip()
        answers = list(item.get("answers") or [])
        gold = (item.get("correct_answer") or "").strip()

        if len(answers) != 4:
            skipped_wrong_arity += 1
            continue
        answers_norm = [(a or "").strip() for a in answers]
        if gold not in answers_norm:
            skipped_no_match += 1
            continue

        qid = f"q{i}"
        queries_rows.append({"_id": qid, "text": question})

        gold_pos = answers_norm.index(gold)
        cand_ids: list[str] = []
        for j, ans in enumerate(answers_norm):
            doc_id = f"{qid}_a{j}"
            corpus_rows.append({"_id": doc_id, "title": "", "text": ans})
            cand_ids.append(doc_id)

        qrels_rows.append(
            {"query-id": qid, "corpus-id": cand_ids[gold_pos], "score": 1}
        )
        top_ranked_rows.append({"query-id": qid, "corpus-ids": cand_ids})

        ctx = item.get("cultural_context")
        if ctx:
            culture_counts[ctx] += 1
        diff = item.get("difficulty")
        if diff:
            difficulty_counts[diff] += 1

    stats = {
        "skipped_no_match": skipped_no_match,
        "skipped_wrong_arity": skipped_wrong_arity,
        "culture": dict(culture_counts),
        "difficulty": dict(difficulty_counts),
    }

    corpus = Dataset.from_list(corpus_rows)
    queries = Dataset.from_list(queries_rows)
    qrels = Dataset.from_list(qrels_rows)
    top_ranked = Dataset.from_list(top_ranked_rows)
    return corpus, queries, qrels, top_ranked, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_reranking", action="store_true")
    parser.add_argument("--skip_retrieval", action="store_true")
    args = parser.parse_args()

    print(f"Loading {SOURCE_REPO}:{SOURCE_CONFIG} ({SOURCE_SPLIT}) ...")
    ds = load_dataset(SOURCE_REPO, SOURCE_CONFIG, split=SOURCE_SPLIT)
    print(f"Loaded {len(ds):,} rows (columns: {ds.column_names})")

    corpus, queries, qrels, top_ranked, stats = _build_beir(ds)
    print_beir_stats("WWTBM-Ro", corpus, queries, qrels, top_ranked, stats)

    if args.dry_run:
        return
    if not args.skip_reranking:
        sha = push_beir_repo(corpus, queries, qrels, TARGET_REPO, top_ranked=top_ranked)
        print(f"\nReranking pin: revision = {sha!r}")
    if not args.skip_retrieval:
        sha_r = push_beir_repo(corpus, queries, qrels, RETRIEVAL_REPO, top_ranked=None)
        print(f"\nRetrieval pin: revision = {sha_r!r}")


if __name__ == "__main__":
    main()
