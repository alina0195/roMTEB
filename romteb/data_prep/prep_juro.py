"""Prepare JuRo legal MCQ into BEIR-style reranking format for RoMTEB.

Expected JuRo tree:
  JuRo/
    train_set/
    validation_set/
    test_set/

The reranking benchmark is built from `test_set` only:
  - queries: one legal exam question
  - corpus: answer options A-E for that question
  - qrels: one or more correct options
  - top_ranked: per-query candidate pool (options for that question only)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from romteb._hf_datasets import Dataset

from romteb.data_prep._beir_io import print_beir_stats, push_beir_repo


TARGET_REPO = "alina0195/romteb-juro-legal-reranking"
RETRIEVAL_REPO = "alina0195/romteb-juro-legal-retrieval"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_source_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"JuRo source dir not found: {p}")

    candidates = [
        WORKSPACE_ROOT / "datasets" / "JuRo",
        WORKSPACE_ROOT / "tmp_graf_repo" / "JuRo",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "JuRo source directory not found. Pass --source_dir or place JuRo under "
        f"{candidates[0]}."
    )


def _extract_options(options_blob: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in (options_blob or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-E])[\.\)]\s*(.+)$", line)
        if not m:
            continue
        letter, text = m.group(1), m.group(2).strip()
        if text:
            out.append((letter, text))
    seen = set()
    deduped: list[tuple[str, str]] = []
    for letter, text in out:
        if letter in seen:
            continue
        seen.add(letter)
        deduped.append((letter, text))
    return deduped


def _extract_gold_letters(answer_blob: str) -> list[str]:
    return sorted(set(re.findall(r"[A-E]", (answer_blob or "").upper())))


def _iter_test_rows(source_dir: Path):
    test_root = source_dir / "test_set"
    if not test_root.exists():
        raise FileNotFoundError(f"Missing test_set under {source_dir}")
    for csv_path in sorted(test_root.rglob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                yield csv_path, row[0].strip(), row[1].strip(), row[2], row[3]


def _build_beir(source_dir: Path):
    corpus_rows: list[dict] = []
    queries_rows: list[dict] = []
    qrels_rows: list[dict] = []
    top_ranked_rows: list[dict] = []

    skipped = 0
    kept = 0
    for idx, (_path, subject, question, options_blob, answer_blob) in enumerate(
        _iter_test_rows(source_dir)
    ):
        options = _extract_options(options_blob)
        if len(options) < 2:
            skipped += 1
            continue

        gold_letters = _extract_gold_letters(answer_blob)
        if not gold_letters:
            skipped += 1
            continue

        qid = f"q_{idx}"
        query_text = f"[{subject}] {question}".strip()
        queries_rows.append({"_id": qid, "text": query_text})

        option_map = {letter: text for letter, text in options}
        cand_ids: list[str] = []
        for letter, text in options:
            doc_id = f"{qid}_{letter}"
            corpus_rows.append({"_id": doc_id, "title": "", "text": text})
            cand_ids.append(doc_id)

        for letter in gold_letters:
            if letter not in option_map:
                continue
            qrels_rows.append({"query-id": qid, "corpus-id": f"{qid}_{letter}", "score": 1})

        if not any(r["query-id"] == qid for r in qrels_rows):
            skipped += 1
            continue

        top_ranked_rows.append({"query-id": qid, "corpus-ids": cand_ids})
        kept += 1

    stats = {"kept": kept, "skipped": skipped}
    return (
        Dataset.from_list(corpus_rows),
        Dataset.from_list(queries_rows),
        Dataset.from_list(qrels_rows),
        Dataset.from_list(top_ranked_rows),
        stats,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=None, help="Path to JuRo root directory")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_reranking", action="store_true")
    parser.add_argument("--skip_retrieval", action="store_true")
    args = parser.parse_args()

    source_dir = _resolve_source_dir(args.source_dir)
    print(f"Using JuRo source: {source_dir}")

    corpus, queries, qrels, top_ranked, stats = _build_beir(source_dir)
    print_beir_stats("JuRo reranking/retrieval", corpus, queries, qrels, top_ranked, stats)

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
