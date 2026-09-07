"""Prepare CLiPS MQA Romanian CQA as a BEIR retrieval eval split.

Source: clips/mqa config ``ro-cqa-question`` (~93k community Q&A rows, train-only).
The Hub repo still ships a loading script (``mqa.py``), which ``datasets>=4``
refuses. Prep loads the auto-converted parquet under
``refs/convert/parquet/ro-cqa-question/``.

Do **not** dump ``ro-all-question`` (mostly FAQ, overlaps WebFAQRetrieval).

Each kept CQA thread becomes:
  query  = question title (+ body when it adds content)
  corpus = answer texts (content-hashed IDs; duplicates collapse)
  qrels  = accepted answers (fallback: all usable answers)

There is no official test split. We:
  - drop empty / spam / English-looking rows
  - cap test queries per web domain (FAQ-style duplication)
  - hold out ``--max_test_queries`` hash-stable questions
  - index **all** filtered CQA answers as the corpus (full retrieval, no top_ranked)

Output: alina0195/romteb-mqa-ro-cqa-retrieval
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict

from romteb._hf_datasets import Dataset, load_dataset
from romteb.data_prep._beir_io import print_beir_stats, push_beir_repo


SOURCE_REPO = "clips/mqa"
SOURCE_CONFIG = "ro-cqa-question"
# datasets>=4 dropped loading scripts. Hub parquet conversion:
SOURCE_PARQUET = (
    "hf://datasets/clips/mqa@refs/convert/parquet/"
    "ro-cqa-question/train/*.parquet"
)
TARGET_REPO = "alina0195/romteb-mqa-ro-cqa-retrieval"

MIN_QUERY_CHARS = 12
MIN_ANSWER_CHARS = 24
MAX_QUERY_CHARS = 2000
MAX_ANSWER_CHARS = 4000

_DIA = re.compile(r"[ăâîșțşţĂÂÎȘȚŞŢ]")
_RO = re.compile(
    r"\b(și|si|pentru|este|sunt|să|sa|cum|cine|unde|când|cand|"
    r"dacă|daca|trebuie|poate|acest|aceasta|această|aceasta|"
    r"întreb|intreb|răspuns|raspuns|mulțumesc|multumesc|"
    r"vreau|niște|niste|fără|fara|către|catre|dece|de ce|"
    r"care|unui|unei|acestui|acestei|doar|mai|cât|cat)\b",
    re.IGNORECASE,
)
_EN = re.compile(
    r"\b(the|and|what|how|this|that|with|from|exam|study|guide|"
    r"chapter|which|your|please|module|review|sheet)\b",
    re.IGNORECASE,
)


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _norm(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _is_repetition_spam(title: str, body: str) -> bool:
    t = title.lower().strip()
    b = body.lower().strip()
    if not t:
        return True
    toks = t.split()
    if toks and len(set(toks)) == 1 and len(toks) <= 6:
        return True
    if b and t and b.count(t) >= 2 and len(t) < 60:
        return True
    return False


def _looks_romanian(text: str) -> bool:
    if _DIA.search(text):
        return True
    ro_hits = len(_RO.findall(text))
    en_hits = len(_EN.findall(text))
    return ro_hits >= 2 and en_hits <= ro_hits


def _query_text(title: str, body: str) -> str:
    if _is_repetition_spam(title, body):
        return title if title else body
    if title and body and body.lower() != title.lower():
        return f"{title}\n{body}"[:MAX_QUERY_CHARS]
    return (title or body)[:MAX_QUERY_CHARS]


def _kept_answers(raw_answers) -> list[str]:
    if not raw_answers:
        return []
    usable: list[str] = []
    accepted: list[str] = []
    seen: set[str] = set()
    for ans in raw_answers:
        if not isinstance(ans, dict):
            continue
        text = _norm(ans.get("text"))[:MAX_ANSWER_CHARS]
        if len(text) < MIN_ANSWER_CHARS:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        usable.append(text)
        if ans.get("is_accepted"):
            accepted.append(text)
    return accepted or usable


def _load_mqa_cqa():
    """Load ro-cqa-question without executing the deprecated mqa.py script."""
    print(f"Loading {SOURCE_REPO}:{SOURCE_CONFIG} from parquet convert ...")
    return load_dataset("parquet", data_files=SOURCE_PARQUET, split="train")


def _collect(max_test_queries: int, max_per_domain: int) -> tuple[Dataset, Dataset, Dataset, dict]:
    ds = _load_mqa_cqa()
    print(f"  {len(ds):,} raw CQA rows")

    skipped = Counter()
    threads: list[dict] = []
    seen_q: set[str] = set()
    corpus_by_id: dict[str, str] = {}

    for row in ds:
        title = _norm(row.get("name"))
        body = _norm(row.get("text"))
        query = _query_text(title, body).strip()
        if len(query) < MIN_QUERY_CHARS:
            skipped["short_query"] += 1
            continue
        if _is_repetition_spam(title, body) and not _looks_romanian(query):
            skipped["spam"] += 1
            continue
        if not _looks_romanian(query):
            skipped["not_romanian"] += 1
            continue
        golds = _kept_answers(row.get("answers"))
        if not golds:
            skipped["no_answer"] += 1
            continue
        golds = [g for g in golds if g.casefold() != query.casefold()]
        if not golds:
            skipped["answer_eq_query"] += 1
            continue
        qkey = query.casefold()
        if qkey in seen_q:
            skipped["dup_query"] += 1
            continue
        seen_q.add(qkey)
        domain = _norm(row.get("domain")) or "unknown"
        src_id = _norm(row.get("id")) or query
        threads.append(
            {"src_id": src_id, "query": query, "domain": domain, "golds": golds}
        )
        for ans in golds:
            corpus_by_id.setdefault(_stable_id("a", ans), ans)

    # Hash-stable test selection with per-domain cap.
    ranked = sorted(threads, key=lambda t: hashlib.md5(t["query"].encode("utf-8")).hexdigest())
    domain_counts: dict[str, int] = defaultdict(int)
    selected: list[dict] = []
    for rec in ranked:
        if len(selected) >= max_test_queries:
            break
        if domain_counts[rec["domain"]] >= max_per_domain:
            continue
        domain_counts[rec["domain"]] += 1
        selected.append(rec)

    queries_rows = []
    qrels_rows = []
    for rec in selected:
        qid = _stable_id("q", rec["src_id"])
        queries_rows.append({"_id": qid, "text": rec["query"]})
        for ans in rec["golds"]:
            did = _stable_id("a", ans)
            qrels_rows.append({"query-id": qid, "corpus-id": did, "score": 1})

    corpus = Dataset.from_list(
        [{"_id": did, "title": "", "text": text} for did, text in corpus_by_id.items()]
    )
    queries = Dataset.from_list(queries_rows)
    qrels = Dataset.from_list(qrels_rows)
    stats = {
        "raw_rows": len(ds),
        "valid_threads": len(threads),
        "skipped": dict(skipped),
        "test_queries": len(queries_rows),
        "test_domains": len(domain_counts),
        "mean_golds": (
            sum(len(r["golds"]) for r in selected) / max(len(selected), 1)
        ),
        "top_test_domains": dict(Counter(domain_counts).most_common(8)),
    }
    return corpus, queries, qrels, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_test_queries", type=int, default=3000)
    parser.add_argument(
        "--max_per_domain",
        type=int,
        default=40,
        help="Max test queries from one web domain (anti-duplication).",
    )
    args = parser.parse_args()

    corpus, queries, qrels, stats = _collect(
        max_test_queries=args.max_test_queries,
        max_per_domain=args.max_per_domain,
    )
    print_beir_stats("MQA RO CQA", corpus, queries, qrels, extra=stats)
    if len(queries) == 0:
        raise SystemExit("No test queries after filtering.")

    if args.dry_run:
        return
    sha = push_beir_repo(
        corpus,
        queries,
        qrels,
        TARGET_REPO,
        top_ranked=None,
        commit_message="RoMTEB MQA Romanian CQA retrieval (filtered hold-out)",
    )
    print(f"\nPin: revision = {sha!r}")


if __name__ == "__main__":
    main()
