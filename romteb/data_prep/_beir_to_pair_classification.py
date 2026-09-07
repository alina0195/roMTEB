"""Convert BEIR-style reranking datasets to MTEB PairClassification format."""

from __future__ import annotations

from collections import Counter
from statistics import mean

from romteb._hf_datasets import Dataset, load_dataset


def load_beir_reranking(repo_id: str):
    """Load corpus / queries / qrels / top_ranked configs from a HF reranking repo."""
    corpus = load_dataset(repo_id, "corpus", split="corpus")
    queries = load_dataset(repo_id, "queries", split="queries")
    qrels = load_dataset(repo_id, "default", split="test")
    top_ranked = load_dataset(repo_id, "top_ranked", split="test")
    return corpus, queries, qrels, top_ranked


def beir_reranking_to_pairs(
    corpus,
    queries,
    qrels,
    top_ranked,
) -> tuple[list[str], list[str], list[int], dict[str, int]]:
    """Expand per-query candidate pools into flat (sentence1, sentence2, label) lists."""
    corpus_by_id = {row["_id"]: row["text"] for row in corpus}
    queries_by_id = {row["_id"]: row["text"] for row in queries}
    positives = {
        (row["query-id"], row["corpus-id"])
        for row in qrels
        if int(row.get("score", 0)) > 0
    }

    sentence1: list[str] = []
    sentence2: list[str] = []
    labels: list[int] = []
    skipped = 0

    for row in top_ranked:
        qid = row["query-id"]
        qtext = queries_by_id.get(qid)
        if not qtext:
            skipped += 1
            continue

        cand_ids = list(row.get("corpus-ids") or [])
        if len(cand_ids) < 2:
            skipped += 1
            continue

        has_positive = any((qid, cid) in positives for cid in cand_ids)
        if not has_positive:
            skipped += 1
            continue

        for cid in cand_ids:
            option_text = corpus_by_id.get(cid)
            if not option_text:
                continue
            sentence1.append(qtext)
            sentence2.append(option_text)
            labels.append(1 if (qid, cid) in positives else 0)

    stats = {
        "kept": len(top_ranked) - skipped,
        "skipped": skipped,
        "pairs": len(labels),
        "positives": sum(labels),
    }
    return sentence1, sentence2, labels, stats


def pack_pair_classification(
    sentence1: list[str],
    sentence2: list[str],
    labels: list[int],
) -> Dataset:
    """One HF row per (question, option) pair with scalar columns."""
    return Dataset.from_list(
        [
            {"sentence1": s1, "sentence2": s2, "label": lab}
            for s1, s2, lab in zip(sentence1, sentence2, labels)
        ]
    )


def print_pair_stats(
    name: str,
    stats: dict[str, int],
    sentence1: list[str],
    sentence2: list[str],
    labels: list[int],
) -> None:
    print(f"\n=== {name} ===")
    print(f"  [test] rows={stats['pairs']:,}  columns=['sentence1', 'sentence2', 'label']")
    print(f"    questions kept={stats['kept']:,}  skipped={stats['skipped']:,}")
    print(f"    positives={stats['positives']:,}")
    dist = Counter(labels)
    print(f"    label distribution: {dict(sorted(dist.items()))}")
    for key, values in (("sentence1", sentence1), ("sentence2", sentence2)):
        lengths = [len(v) for v in values if v]
        if lengths:
            print(
                f"    {key}: min={min(lengths)} max={max(lengths)} "
                f"mean={mean(lengths):.1f} chars"
            )
