"""RoMTEB benchmark assembly.

Two lists are exposed:

  ROMTEB_REUSED   - Tasks already in MTEB with Romanian subsets.
  ROMTEB_CUSTOM   - Task classes defined in `romteb/tasks/`.
  ROMTEB_TASKS    - The concatenation; this is what the runner uses.
"""

from __future__ import annotations

import sys
from typing import Iterable

import mteb

from romteb.tasks.classification import (
    HateSpeechROClassification,
    HistNERoMentionClassification,
    REDv2EmotionClassification,
    RoABSAClassification,
    RoMathDomainClassification,
    RoOffenseClassification,
    SaRoCoClassification,
    SciTechBanROClassification,
)
from romteb.tasks.pair_classification import RoNLIPairClassification
from romteb.tasks.reranking import (
    GrileGrammarReranking,
    JuRoLegalExamReranking,
    RoMedQAv2Reranking,
    WWTBMRoQAReranking,
)
from romteb.tasks.retrieval import (
    MQARoCQARetrieval,
    RoDTALLawsRetrieval,
)
from romteb.eval_config import apply_eval_config
from romteb.tasks.sts import RoSTS


def _instantiate(cls):
    """Best-effort instantiation; print a warning if a task fails to load."""
    try:
        return cls()
    except Exception as exc:  # pragma: no cover - depends on HF availability
        print(f"[romteb] skip {cls.__name__}: {exc}", file=sys.stderr)
        return None


_CUSTOM_CLASSES = [
    # Classification (frozen embedder + logistic regression probe)
    RoABSAClassification,
    RoOffenseClassification,
    HateSpeechROClassification,
    REDv2EmotionClassification,
    RoMathDomainClassification,
    SciTechBanROClassification,
    SaRoCoClassification,
    HistNERoMentionClassification,
    # Pair Classification (cosine AP, no LR)
    RoNLIPairClassification,
    # STS
    RoSTS,
    # Retrieval: open corpus (not MCQ). MS MARCO is held out for training.
    RoDTALLawsRetrieval,
    MQARoCQARetrieval,
    # Reranking: MCQ option pool (JuRo / WWTBM / RoMedQA / GRILE).
    JuRoLegalExamReranking,
    WWTBMRoQAReranking,
    RoMedQAv2Reranking,
    GrileGrammarReranking,
    # NOTE: the four full-option-bank MCQ retrieval variants (Grile / JuRo /
    # WWTBM / RoMedQA *Retrieval*) were removed in Sep 2026 after the audit
    # (docs/task_audit.md). Three of them collapsed to p90 nDCG@10 < 0.10
    # because the option-bank corpus has duplicate surface forms, and the
    # WWTBM variant was driven by a single outlier. Use the *Reranking
    # counterparts instead — the official MCQ protocol.
    # Clustering is out of this stage (RORetrieval unpinned; SIB200 skipped).
]

ROMTEB_CUSTOM = [t for t in (_instantiate(c) for c in _CUSTOM_CLASSES) if t is not None]


REUSED_TASK_NAMES: list[str] = [
    # Classification
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "SIB200Classification",
    # Moroco.v2 dropped (obsolete)
    "RomanianReviewsSentiment.v2",
    "RomanianSentimentClassification.v2",
    # Retrieval
    "WebFAQRetrieval",
    "WikipediaRetrievalMultilingual",
    "XQuADRetrieval",
    # BitextMining (cross-lingual section, excluded from Overall)
    "NTREXBitextMining",
    "Tatoeba",
    "IWSLT2017BitextMining",
]

_DEFAULT_SUBSET_TASKS = {
    "RomanianReviewsSentiment.v2",
    "RomanianSentimentClassification.v2",
}


def _get_tasks_safe(name: str, languages: list[str]) -> list:
    """Call mteb.get_tasks, catching both TypeError and ValueError."""
    for lang in languages:
        try:
            hits = mteb.get_tasks(tasks=[name], languages=[lang])
            if hits:
                return hits
        except (TypeError, ValueError):
            pass
    return []


def _load_reused(names: Iterable[str], languages=("ron", "ron-Latn", "ron_Latn")) -> list:
    """Resolve reused task names via mteb.get_tasks; tolerate missing ones."""
    available = []
    for name in names:
        if name in _DEFAULT_SUBSET_TASKS:
            try:
                hits = mteb.get_tasks(tasks=[name])
            except Exception as exc:
                print(f"[romteb] cannot load {name}: {exc}", file=sys.stderr)
                hits = []
        else:
            hits = _get_tasks_safe(name, list(languages))
            if not hits:
                print(f"[romteb] reused task not found: {name}", file=sys.stderr)
        available.extend(hits)

    seen, deduped = set(), []
    for t in available:
        key = getattr(getattr(t, "metadata", None), "name", repr(t))
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped


ROMTEB_REUSED = apply_eval_config(_load_reused(REUSED_TASK_NAMES))

ROMTEB_TASKS = apply_eval_config(ROMTEB_REUSED + ROMTEB_CUSTOM)


def get_benchmark():
    """Return the MTEB v2 Benchmark wrapper if available, else the task list."""
    Benchmark = getattr(mteb, "Benchmark", None)
    if Benchmark is None:
        return ROMTEB_TASKS
    try:
        return Benchmark(
            name="RoMTEB",
            description="Romanian Massive Text Embedding Benchmark",
            tasks=ROMTEB_TASKS,
        )
    except TypeError:
        return Benchmark(tasks=ROMTEB_TASKS)


__all__ = [
    "ROMTEB_TASKS",
    "ROMTEB_REUSED",
    "ROMTEB_CUSTOM",
    "REUSED_TASK_NAMES",
    "get_benchmark",
]
