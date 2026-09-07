"""Prepare RoMedQA_v2 MCQ as BEIR-style reranking for RoMTEB."""

from __future__ import annotations

import argparse
import re

from romteb._hf_datasets import Dataset, load_dataset

from romteb.data_prep._beir_io import print_beir_stats, push_beir_repo


SOURCE_REPO = "craciuncg/RoMedQA_v2"
TARGET_REPO = "alina0195/romteb-romedqa-v2-reranking"
RETRIEVAL_REPO = "alina0195/romteb-romedqa-v2-retrieval"


def _split_question_and_options(question_text: str) -> tuple[str, dict[str, str]]:
    lines = [ln.strip() for ln in (question_text or "").splitlines() if ln.strip()]
    stem_lines: list[str] = []
    options: dict[str, str] = {}
    found_options = False

    for ln in lines:
        m = re.match(r"^([1-5])\.\s*(.+)$", ln)
        if m:
            found_options = True
            options[m.group(1)] = m.group(2).strip()
        elif not found_options:
            stem_lines.append(ln)

    stem = " ".join(stem_lines).strip()
    if not stem:
        stem = (question_text or "").strip()
    return stem, options


def _extract_correct_choices(answer_value) -> list[str]:
    text = str(answer_value)
    return sorted(set(re.findall(r"[1-5]", text)))


def _build_beir(ds: Dataset):
    corpus_rows: list[dict] = []
    queries_rows: list[dict] = []
    qrels_rows: list[dict] = []
    top_ranked_rows: list[dict] = []
    skipped = 0

    for i, row in enumerate(ds):
        stem, options = _split_question_and_options((row.get("question") or "").strip())
        if not stem or len(options) < 2:
            skipped += 1
            continue

        correct = [c for c in _extract_correct_choices(row.get("answer")) if c in options]
        if not correct:
            skipped += 1
            continue

        qid = f"q_{i}"
        queries_rows.append({"_id": qid, "text": stem})

        cand_ids = []
        for choice in sorted(options.keys()):
            doc_id = f"{qid}_c{choice}"
            corpus_rows.append({"_id": doc_id, "title": "", "text": options[choice]})
            cand_ids.append(doc_id)

        for choice in correct:
            qrels_rows.append({"query-id": qid, "corpus-id": f"{qid}_c{choice}", "score": 1})

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
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_reranking", action="store_true")
    parser.add_argument("--skip_retrieval", action="store_true")
    args = parser.parse_args()

    print(f"Loading {SOURCE_REPO} (split=test)")
    ds = load_dataset(SOURCE_REPO, split="test")
    corpus, queries, qrels, top_ranked, stats = _build_beir(ds)
    print_beir_stats("RoMedQA_v2", corpus, queries, qrels, top_ranked, stats)

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
