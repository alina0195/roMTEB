"""Audit MCQ option corpora: exact-text duplicates and unevaluable golds.

Two independent problems:

1. **Cross-question duplicates** (Retrieval only). The same option text is
   gold for one question and a distractor for another, so cosine ties are
   decided by sort order. Reranking pools are per-question, so this does
   not affect the ranking metric.

2. **Meta-golds** (Retrieval *and* Reranking). If the correct option is
   "Toate variantele de mai sus" / "Nicio variantă" / "All of the above",
   a frozen embedder cannot match the question to the answer by meaning.
   Those items should be dropped from any protocol that indexes option
   text (pending cleanup for the Retrieval variants; same filter applies
   to Reranking if the rate is high).

USAGE:
    ./apptainer-exec-romteb.sh scripts/mcq_duplicate_stats.py
    ./apptainer-exec-romteb.sh scripts/mcq_duplicate_stats.py --reranking_only
    ./apptainer-exec-romteb.sh scripts/mcq_duplicate_stats.py \\
        --out_csv results/mcq_option_audit.csv \\
        --out_drop results/mcq_meta_gold_drop.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from romteb._hf_datasets import load_dataset

RETRIEVAL_REPOS = [
    ("JuRoLegalExamRetrieval", "alina0195/romteb-juro-legal-retrieval"),
    ("WWTBMRoQARetrieval", "alina0195/romteb-wwtbm-ro-retrieval"),
    ("RoMedQAv2Retrieval", "alina0195/romteb-romedqa-v2-retrieval"),
    ("GrileGrammarRetrieval", "alina0195/romteb-grile-retrieval"),
]

# Reranking repos still have a full option corpus; report both.
RERANKING_REPOS = [
    ("JuRoLegalExamReranking", "alina0195/romteb-juro-legal-reranking"),
    ("WWTBMRoQAReranking", "alina0195/romteb-wwtbm-ro"),
    ("RoMedQAv2Reranking", "alina0195/romteb-romedqa-v2-reranking"),
    ("GrileGrammarReranking", "alina0195/romteb-grile-reranking"),
]

_WS = re.compile(r"\s+")

# Gold / option text that an embedder cannot score by meaning.
_META_PATTERNS = (
    re.compile(r"\btoate (cele|variantele|raspunsurile|afirmatiile)( de mai (sus|inainte))?\b"),
    re.compile(r"\btoate (de mai sus|optiunile)\b"),
    re.compile(r"\b(nici ?o|nicio) varianta\b"),
    re.compile(r"\bnici(o|una|unul) dintre (variante|raspunsuri|optiuni)\b"),
    re.compile(r"\bnici(o|una|unul) dintre cele de mai sus\b"),
    re.compile(r"\bnone of the (above|options|answers)\b"),
    re.compile(r"\ball of the (above|options|answers)\b"),
    re.compile(r"\bboth of the above\b"),
    re.compile(r"\bboth [a-e] and [a-e]\b"),
    re.compile(r"^[a-e](,? (si|and|&) [a-e])+$"),
)


def _norm(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text or "").strip().lower()
    return _WS.sub(" ", folded)


def _fold(text: str) -> str:
    nkd = unicodedata.normalize("NFKD", _norm(text))
    return "".join(c for c in nkd if not unicodedata.combining(c))


def is_meta_option(text: str) -> bool:
    folded = _fold(text)
    return any(p.search(folded) for p in _META_PATTERNS)


def _load_corpus(repo_id: str):
    try:
        return load_dataset(repo_id, "corpus", split="corpus")
    except Exception:
        try:
            return load_dataset(repo_id, "corpus")["corpus"]
        except Exception as exc:
            print(f"  skip corpus {repo_id}: {exc}")
            return None


def _load_qrels(repo_id: str):
    for cfg, split in (("default", "test"), ("qrels", "test"), ("default", "validation")):
        try:
            ds = load_dataset(repo_id, cfg, split=split)
            return ds
        except Exception:
            continue
    print(f"  skip qrels {repo_id}")
    return None


def _load_top_ranked(repo_id: str):
    for cfg in ("top_ranked", "default"):
        try:
            ds = load_dataset(repo_id, cfg, split="test")
            if "corpus-ids" in ds.column_names or "corpus_ids" in ds.column_names:
                return ds
        except Exception:
            continue
    return None


def _doc_text(row: dict) -> str:
    title = row.get("title") or ""
    text = row.get("text") or row.get("content") or ""
    return f"{title} {text}".strip()


def _stats(name: str, repo_id: str) -> dict | None:
    print(f"\n=== {name} ({repo_id}) ===")
    corpus = _load_corpus(repo_id)
    if corpus is None:
        return None
    id_key = "_id" if "_id" in corpus.column_names else "id"
    id_to_text: dict[str, str] = {}
    id_to_raw: dict[str, str] = {}
    for row in corpus:
        doc_id = str(row[id_key])
        raw = _doc_text(row)
        id_to_raw[doc_id] = raw
        id_to_text[doc_id] = _norm(raw)
    counts = Counter(id_to_text.values())
    n_docs = len(id_to_text)
    n_dup_docs = sum(c for c in counts.values() if c >= 2)
    pct_docs = 100.0 * n_dup_docs / n_docs if n_docs else 0.0
    n_meta_docs = sum(1 for raw in id_to_raw.values() if is_meta_option(raw))
    print(f"  corpus={n_docs:,}  unique_texts={len(counts):,}  "
          f"docs with duplicate text={n_dup_docs:,} ({pct_docs:.1f}%)")
    print(f"  meta-option docs (all of the above / none / A și B)="
          f"{n_meta_docs:,} ({100.0 * n_meta_docs / n_docs if n_docs else 0:.1f}%)")

    qrels = _load_qrels(repo_id)
    top_ranked = _load_top_ranked(repo_id)
    pool_by_q: dict[str, list[str]] = {}
    if top_ranked is not None:
        qid_col = "query-id" if "query-id" in top_ranked.column_names else "query_id"
        ids_col = "corpus-ids" if "corpus-ids" in top_ranked.column_names else "corpus_ids"
        for row in top_ranked:
            pool_by_q[str(row[qid_col])] = [str(x) for x in row[ids_col]]

    audit_rows: list[dict] = []
    pct_queries = None
    n_queries = 0
    n_hit = 0
    n_meta_gold = 0
    if qrels is not None:
        qid_col = "query-id" if "query-id" in qrels.column_names else "query_id"
        cid_col = "corpus-id" if "corpus-id" in qrels.column_names else "corpus_id"
        gold: dict[str, list[str]] = defaultdict(list)
        for row in qrels:
            gold[str(row[qid_col])].append(str(row[cid_col]))
        text_to_ids: dict[str, set[str]] = defaultdict(set)
        for doc_id, text in id_to_text.items():
            text_to_ids[text].add(doc_id)
        n_queries = len(gold)
        for qid, gold_ids in gold.items():
            gold_texts = {id_to_text[g] for g in gold_ids if g in id_to_text}
            gold_raws = [id_to_raw[g] for g in gold_ids if g in id_to_raw]
            collision = False
            for gt in gold_texts:
                others = text_to_ids[gt] - set(gold_ids)
                if others:
                    collision = True
                    break
            if collision:
                n_hit += 1
            gold_is_meta = any(is_meta_option(t) for t in gold_raws)
            if gold_is_meta:
                n_meta_gold += 1
            pool_ids = pool_by_q.get(qid, gold_ids)
            n_meta_in_pool = sum(
                1 for d in pool_ids if d in id_to_raw and is_meta_option(id_to_raw[d])
            )
            gold_preview = " | ".join(_norm(t) for t in gold_raws)[:200]
            audit_rows.append(
                {
                    "task": name,
                    "repo": repo_id,
                    "query_id": qid,
                    "gold_ids": " ".join(gold_ids),
                    "gold_text": gold_preview,
                    "gold_has_corpus_duplicate": int(collision),
                    "gold_is_meta": int(gold_is_meta),
                    "n_meta_options_in_pool": n_meta_in_pool,
                    "drop_recommended": int(gold_is_meta),
                }
            )
        pct_queries = 100.0 * n_hit / n_queries if n_queries else 0.0
        pct_meta = 100.0 * n_meta_gold / n_queries if n_queries else 0.0
        print(
            f"  queries={n_queries:,}  gold has exact duplicate in corpus="
            f"{n_hit:,} ({pct_queries:.1f}%)"
        )
        print(
            f"  queries whose GOLD is a meta-option="
            f"{n_meta_gold:,} ({pct_meta:.1f}%)  ← pending Retrieval cleanup"
        )
        print("  most duplicated option texts:")
        for text, c in counts.most_common(8):
            if c < 2:
                break
            flag = "  [META]" if is_meta_option(text) else ""
            preview = text[:80] + ("…" if len(text) > 80 else "")
            print(f"    {c:4d}×  {preview}{flag}")
    return {
        "task": name,
        "repo": repo_id,
        "n_corpus": n_docs,
        "pct_dup_docs": round(pct_docs, 2),
        "n_queries": n_queries,
        "pct_gold_collision": None if pct_queries is None else round(pct_queries, 2),
        "n_meta_gold": n_meta_gold,
        "pct_meta_gold": (
            None if n_queries == 0 else round(100.0 * n_meta_gold / n_queries, 2)
        ),
        "audit_rows": audit_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reranking_only",
        action="store_true",
        help="Skip held-out retrieval repos (they may not be on the Hub).",
    )
    parser.add_argument(
        "--out_csv",
        default="results/mcq_option_audit.csv",
        help="Per-query audit (duplicates + meta-golds).",
    )
    parser.add_argument(
        "--out_drop",
        default="results/mcq_meta_gold_drop.csv",
        help="Query ids whose gold is a meta-option (pending Retrieval cleanup).",
    )
    args = parser.parse_args()
    repos = list(RERANKING_REPOS)
    if not args.reranking_only:
        repos = RETRIEVAL_REPOS + repos
    rows = []
    audit: list[dict] = []
    for name, repo in repos:
        row = _stats(name, repo)
        if row:
            audit.extend(row.pop("audit_rows", []))
            rows.append(row)
    if not rows:
        raise SystemExit("no MCQ corpora could be loaded")

    print("\n=== Decision table (paste into the evidence note) ===")
    print(f"{'task':<28} {'n_q':>6} {'dup_gold%':>10} {'meta_gold%':>11}")
    for row in rows:
        dup = "-" if row["pct_gold_collision"] is None else f"{row['pct_gold_collision']:.1f}"
        meta = "-" if row["pct_meta_gold"] is None else f"{row['pct_meta_gold']:.1f}"
        print(f"{row['task']:<28} {row['n_queries']:>6} {dup:>10} {meta:>11}")

    print(
        "\nHow to read this:\n"
        "  dup_gold%  high on *Retrieval* → keep MCQ Retrieval out of the benchmark\n"
        "             (already the v1 decision). Reranking is unaffected.\n"
        "  meta_gold% questions whose correct option is 'toate variantele' /\n"
        "             'nicio variantă' / 'all of the above' / 'A și B'. Drop these\n"
        "             from Retrieval (pending cleanup) and, if the rate is high,\n"
        "             from Reranking too — a frozen embedder cannot score them.\n"
        "  Do not drop items that only have a meta *distractor* if the gold is\n"
        "             a real answer."
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if audit:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(audit[0].keys()))
            writer.writeheader()
            writer.writerows(audit)
        print(f"\nWrote {out_csv} ({len(audit)} queries)")

    drop = [r for r in audit if r["drop_recommended"]]
    out_drop = Path(args.out_drop)
    out_drop.parent.mkdir(parents=True, exist_ok=True)
    with out_drop.open("w", newline="", encoding="utf-8") as fh:
        fields = ["task", "query_id", "gold_text"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in drop:
            writer.writerow({k: row[k] for k in fields})
    print(f"Wrote {out_drop} ({len(drop)} queries to drop)")


if __name__ == "__main__":
    main()
