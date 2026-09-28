"""Non-model baselines for RoNLIPairClassification.

Reports average precision (AP) of two surface-cue rankings on the RoNLI
Entailment-vs-Contrastive test pairs, so their scores can sit next to
the embedding models' ``max_ap``:

    chance          = share of positives
    word_overlap    = Jaccard(tok(s1), tok(s2))
    negation_cue    = 1 if either sentence contains a Romanian negation
                      marker, else 0

Bootstrap 95% CIs are computed by resampling the ranking with
replacement (n=1000 by default).

USAGE:
    python scripts/ronli_baselines.py
    python scripts/ronli_baselines.py --hf_revision <sha>  # pin a revision
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from romteb.data_prep._common import read_revision

_HF_PATH = "alina0195/romteb-ronli"
CONTRASTIVE, ENTAILMENT = 0, 1

# Everyday Romanian negation markers. Lower-cased match.
NEGATION_TOKENS = {"nu", "n-", "ne", "nici", "niciodata", "niciodată", "fara", "fără"}

_TOK_RE = re.compile(r"[\wăâîșşțţ]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOK_RE.findall(text.lower())


def word_overlap(s1: str, s2: str) -> float:
    """Jaccard token overlap, in [0, 1]."""
    a, b = set(_tokens(s1)), set(_tokens(s2))
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def negation_cue(s1: str, s2: str) -> float:
    for text in (s1, s2):
        toks = set(_tokens(text))
        if toks & NEGATION_TOKENS:
            return 1.0
        # detect n- clitic ("nu-l", "n-am")
        if re.search(r"\bn[- ]", text.lower()):
            return 1.0
    return 0.0


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """AP with random tie-breaking; positive label = 1."""
    if labels.size == 0:
        return float("nan")
    jitter = np.random.default_rng(0).uniform(0, 1e-9, size=scores.size)
    order = np.argsort(-(scores + jitter))
    labels = labels[order]
    tp_cum = np.cumsum(labels)
    denom = np.arange(1, labels.size + 1)
    precisions = tp_cum / denom
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    return float((precisions * labels).sum() / n_pos)


def bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray, *, n_boot: int = 1000
) -> tuple[float, float, float]:
    n = labels.size
    point = average_precision(scores, labels)
    rng = np.random.default_rng(42)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = average_precision(scores[idx], labels[idx])
    lo, hi = np.percentile(draws[np.isfinite(draws)], [2.5, 97.5])
    return point, float(lo), float(hi)


def _load_pairs(revision: str | None, local_json: Path | None) -> list[dict]:
    if local_json:
        return json.loads(local_json.read_text(encoding="utf-8"))
    from datasets import load_dataset
    rev = revision or read_revision(_HF_PATH)
    ds = load_dataset(_HF_PATH, split="test", revision=rev)
    return [dict(x) for x in ds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_revision", default=None,
                        help="Override the revision pin in romteb/tasks/_revisions.json.")
    parser.add_argument("--local_json", type=Path, default=None,
                        help="Read pairs from a local JSON list instead of the Hub.")
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    rows = _load_pairs(args.hf_revision, args.local_json)
    binary = [r for r in rows if int(r["label"]) in (ENTAILMENT, CONTRASTIVE)]
    labels = np.asarray([1 if int(r["label"]) == ENTAILMENT else 0 for r in binary])
    n_pos = int(labels.sum())
    n = labels.size
    counts = Counter(int(r["label"]) for r in binary)

    print(f"RoNLIPairClassification test pairs: n={n}  "
          f"(Entailment={counts.get(ENTAILMENT, 0)}, "
          f"Contrastive={counts.get(CONTRASTIVE, 0)})")

    chance = n_pos / n if n else float("nan")
    print(f"chance          AP = {chance:.4f}  (share of positives)")

    overlap_scores = np.asarray([
        word_overlap(r["sentence1"], r["sentence2"]) for r in binary
    ])
    mu, lo, hi = bootstrap_ci(overlap_scores, labels, n_boot=args.n_boot)
    print(f"word_overlap    AP = {mu:.4f}  [{lo:.4f}, {hi:.4f}]")

    neg_scores = np.asarray([
        negation_cue(r["sentence1"], r["sentence2"]) for r in binary
    ])
    mu, lo, hi = bootstrap_ci(neg_scores, labels, n_boot=args.n_boot)
    print(f"negation_cue    AP = {mu:.4f}  [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
