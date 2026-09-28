"""RoMTEB evaluation knobs shared by task classes and the aggregator.

Classification protocol
    Default is the full-train class-weighted logistic-regression probe
    (``classification_shots`` returns ``FULL_TRAIN`` / per-task cap and
    ``n_experiments = 1``). Two overrides:

      * ``ROMTEB_K_SHOT=<int>`` forces every classification task to
        that shot count with ``N_EXPERIMENTS`` draws and the MTEB
        default probe — a learning-curve knob.
      * The ``--skip_k8`` flag suppresses the 8-shot sidecar that
        otherwise runs alongside the full-train probe.

    Older MTEB numbers were 8 shots per class × 10 draws; the sidecar
    keeps them for direct comparison.

Reporting domain / dataset
    One primary application domain per task (not the generic ``Written``
    tag). ``TASK_DATASET`` maps tasks to a source corpus so domain and
    type means do not double-count MASSIVE. Official MCQ protocol is
    reranking; full-bank Retrieval variants are diagnostic only
    (``MCQ_RETRIEVAL_TASKS``). See ``docs/evaluation_protocol.md`` §9.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from sklearn.linear_model import LogisticRegression


_K_SHOT_ENV = "ROMTEB_K_SHOT"


def _k_shot_override() -> int | None:
    raw = os.environ.get(_K_SHOT_ENV)
    if raw is None or raw == "":
        return None
    try:
        k = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_K_SHOT_ENV} must be an int, got {raw!r}") from exc
    if k <= 0:
        raise ValueError(f"{_K_SHOT_ENV} must be positive, got {k}")
    return k

# Models skipped from Borda / plots. Nemotron retrieval is still the
# mis-prompted v1 run (prompt-rerun stopped after bge-multilingual-gemma2).
LEADERBOARD_EXCLUDE: set[str] = {
    "nvidia/llama-embed-nemotron-8b",
}

# Keep the MTEB default unless a task is listed below.
DEFAULT_SAMPLES_PER_LABEL = 8   # MTEB default; used by the k8 sidecar
N_EXPERIMENTS = 10              # MTEB default; used by the k8 sidecar

# Official RoMTEB protocol: full-train linear probe.
# FULL_TRAIN is larger than any class count, so _undersample_data keeps every row.
FULL_TRAIN = 10**9

# Optional per-class caps for very large / expensive train splits.
# Only classes above the cap are subsampled; smaller classes stay whole.
FULL_TRAIN_CAPS: dict[str, int] = {
    # "RoNLIClassification": 5000,   # uncomment if pair features get too heavy
}


def make_probe() -> LogisticRegression:
    """Probe for the full-train protocol (a fresh instance per task)."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")

# Rationale lives in docs/evaluation_protocol.md.
CLASSIFICATION_SHOTS: dict[str, int] = {
    # Coarse document labels (binary / few-class).
    "HateSpeechROClassification": 8,
    "SaRoCoClassification": 8,
    "RomanianSentimentClassification.v2": 8,
    "SIB200Classification": 8,
    # MASSIVE already has ~60 intents / ~18 scenarios; 8/class is a large pool.
    "MassiveIntentClassification": 8,
    "MassiveScenarioClassification": 8,
    # Fine-grained multi-class.
    "RoOffenseClassification": 16,          # 5 offense classes
    "REDv2EmotionClassification": 16,       # 7 emotions (projected from multi-label)
    "SciTechBanROClassification": 16,       # sci/tech news labels
    "HistNERoMentionClassification": 16,    # 5 types, span-conditioned
    # Aspect-level: 3 polarities × 15 closed aspects, 1–3 aspects/review typical.
    # 8/class = 24 shots — too few distinct aspects. Neutral is rare in train
    # (~9 rows), so the probe still caps at the available count for that class.
    # Macro-F1 is reported beside accuracy because of that imbalance.
    "RoABSAClassification": 32,
}

# Primary domain for leaderboard slices. One tag per task. Four full-option
# bank MCQ retrieval tasks (Grile / JuRo / WWTBM / RoMedQA Retrieval) were
# removed in Sep 2026 — see docs/task_audit.md.
TASK_REPORTING_DOMAIN: dict[str, str] = {
    # Legal
    "JuRoLegalExamRetrieval": "Legal",
    "RoDTALLawsRetrieval": "Legal",
    # Medical
    "RoMedQAv2Retrieval": "Medical",
    # Academic split: grammar / math / science, plus SIB-200 as general topic.
    "GrileGrammarRetrieval": "Grammar",
    "SciTechBanROClassification": "Science",
    "SIB200Classification": "Academic",
    "SIB200ClusteringS2S": "Academic",
    "RoNewsOutletClusteringP2P": "News",
    "RoNewsTypeClusteringP2P": "News",
    # Reviews / sentiment
    "RoABSAClassification": "Reviews",
    "RomanianSentimentClassification.v2": "Reviews",
    # News / social
    "RoOffenseClassification": "News",
    "HistNERoMentionClassification": "News",
    "HateSpeechROClassification": "Social",
    "SaRoCoClassification": "Social",
    "REDv2EmotionClassification": "Social",
    "MassiveIntentClassification": "Spoken",
    "MassiveScenarioClassification": "Spoken",
    # Culture / encyclopaedic QA
    "WWTBMRoQARetrieval": "Culture",
    "WikipediaRetrievalMultilingual": "Encyclopaedic",
    "WikipediaRerankingMultilingual": "Encyclopaedic",
    "XQuADRetrieval": "Encyclopaedic",
    "BelebeleRetrieval": "Encyclopaedic",
    # Open-domain web IR
    "WebFAQRetrieval": "Web",
    "MQARoCQARetrieval": "Web",
    "RoNLIPairClassification": "Encyclopaedic",
    "RoNLIClassification": "Encyclopaedic",
    "RoSTS": "News",
}

DOMAIN_ORDER = [
    "Legal",
    "Medical",
    "Grammar",
    "Math",
    "Science",
    "Academic",
    "Reviews",
    "News",
    "Social",
    "Spoken",
    "Culture",
    "Encyclopaedic",
    "Web",
]

# Kept as an empty legacy hook for downstream aggregate scripts and any
# lingering imports. The four full-option-bank MCQ retrieval variants
# (Grile / JuRo / WWTBM / RoMedQA) were removed from the benchmark after the
# September 2026 audit (`docs/task_audit.md`): three collapsed to near-zero
# nDCG@10 and the WWTBM variant was driven entirely by one outlier. The
# corresponding data files, HF pins, and Hub repos were removed.
MCQ_RETRIEVAL_TASKS: frozenset[str] = frozenset()

# Retrieval tasks whose BM25 lexical baseline already saturates the metric
# (BM25 ≥ 0.85 and the top of the dense pack is within 0.05 of BM25). These
# stay in the descriptive tables (per-task heatmaps) so the audit finding
# is visible, but are excluded from Overall / type-macro / domain means:
# a saturated task doesn't discriminate between models. See `docs/task_audit.md`.
SATURATED_TASKS: frozenset[str] = frozenset({"XQuADRetrieval"})

# Tasks executed and reported per-task but never counted toward the Overall
# / type-macro / domain aggregation because every model scores at or below
# the empirical random baseline (embedding cosine can't solve the underlying
# task; see `scripts/mcq_multigold_baseline.py` for the baselines).
OVERALL_EXCLUDE_TASKS: frozenset[str] = frozenset({"JuRoLegalExamRetrieval"})

# Source dataset for hierarchical aggregation. Two tasks from the same source
# (MASSIVE intent+scenario) must not be counted twice inside one
# (domain, task-type) mean.
TASK_DATASET: dict[str, str] = {
    "HateSpeechROClassification": "HateSpeech-RO",
    "SaRoCoClassification": "SaRoCo",
    "RomanianSentimentClassification.v2": "RomanianSentiment",
    "SIB200Classification": "SIB-200",
    "SIB200ClusteringS2S": "SIB-200",
    "RoNewsOutletClusteringP2P": "RORetrieval",
    "RoNewsTypeClusteringP2P": "RORetrieval",
    "MassiveIntentClassification": "MASSIVE",
    "MassiveScenarioClassification": "MASSIVE",
    "RoOffenseClassification": "RoOffense",
    "REDv2EmotionClassification": "REDv2",
    "SciTechBanROClassification": "SciTechBanRO",
    "HistNERoMentionClassification": "HistNERo",
    "RoABSAClassification": "RoABSA",
    "RoNLIPairClassification": "RoNLI",
    "RoNLIClassification": "RoNLI",  # same source: never double-count in domain means
    "RoSTS": "RoSTS",
    "WebFAQRetrieval": "WebFAQ",
    "WikipediaRetrievalMultilingual": "Wikipedia",
    "WikipediaRerankingMultilingual": "Wikipedia",  # same source: never double-count
    "XQuADRetrieval": "XQuAD",
    "BelebeleRetrieval": "Belebele",
    "RoDTALLawsRetrieval": "RoD-TAL",
    "MQARoCQARetrieval": "MQA-CQA",
    "JuRoLegalExamRetrieval": "JuRo",
    "WWTBMRoQARetrieval": "WWTBM",
    "RoMedQAv2Retrieval": "RoMedQA",
    "GrileGrammarRetrieval": "GRILE",
    "NTREXBitextMining": "NTREX",
    "Tatoeba": "Tatoeba",
    "IWSLT2017BitextMining": "IWSLT2017",
}

# Primary metric is fixed per MTEB task type. Domain/type means only average
# cells that share this metric. Reranking has per-task overrides below.
CATEGORY_PRIMARY_METRIC: dict[str, str] = {
    "Classification": "accuracy",
    "PairClassification": "cosine_ap",
    "STS": "cosine_spearman",
    "Retrieval": "ndcg_at_10",
    "Reranking": "map_at_1000",
    "BitextMining": "f1",
    "Clustering": "v_measure",
    "Summarization": "cosine_spearman",
}

# Single-gold MCQ: accuracy@1 (Recall@1) is the exam metric. RoMedQA is the
# only multi-relevant pool, so it keeps MAP. Type-macro uses the majority
# metric inside the category (accuracy@1 after JuRo is Overall-excluded);
# category ranking uses Borda so MAP still votes.
TASK_PRIMARY_METRIC: dict[str, str] = {
    "JuRoLegalExamRetrieval": "accuracy",
    "WWTBMRoQARetrieval": "accuracy",
    "GrileGrammarRetrieval": "accuracy",
    "RoMedQAv2Retrieval": "map_at_1000",
}

# Closed-set MCQ random baselines. Values below are empirical mean over
# uniform random permutations (200 trials × n_queries), accounting for the
# actual multi-gold rate in `default` qrels — NOT the naive 1/k / H_k/k
# single-gold estimates. See scripts/mcq_multigold_baseline.py.
#
# Multi-gold rates: JuRo 1975/1730 (14% queries have 2 golds), RoMedQA
# 2911/1699 (mean 1.7 golds/q, pool k=5). GRILE and WWTBM are strict
# single-gold. Consequence: JuRo and RoMedQA random baselines are much
# higher than the naive 1/k, and most published legacy scores sit on top
# of these baselines.
RERANKING_RANDOM_BASELINE: dict[str, dict[str, float]] = {
    "JuRoLegalExamRetrieval": {"k": 3, "accuracy": 0.381, "map_at_1000": 0.639},
    "WWTBMRoQARetrieval": {"k": 4, "accuracy": 0.249, "map_at_1000": 0.520},
    "GrileGrammarRetrieval": {"k": 4, "accuracy": 0.260, "map_at_1000": 0.531},
    "RoMedQAv2Retrieval": {"k": 5, "accuracy": 0.342, "map_at_1000": 0.553},
}

# Classification tasks that also print macro-F1 next to accuracy.
REPORT_MACRO_F1: frozenset[str] = frozenset({"RoABSAClassification", "RoNLIClassification"})


def classification_shots(task_name: str) -> int:
    """Train budget per label.

    Default: full-train (``FULL_TRAIN``) with a per-task cap when the
    train split is expensive. ``ROMTEB_K_SHOT=<int>`` overrides both
    with a fixed shot count for a learning-curve sweep — the caller is
    then also responsible for restoring ``n_experiments`` and the
    default MTEB probe (``apply_eval_config`` does this).
    """
    override = _k_shot_override()
    if override is not None:
        return override
    return FULL_TRAIN_CAPS.get(task_name, FULL_TRAIN)


def reporting_domain(task_name: str) -> str | None:
    return TASK_REPORTING_DOMAIN.get(task_name)


def dataset_of(task_name: str) -> str:
    return TASK_DATASET.get(task_name, task_name)


def metric_of(task_type: str, task_name: str | None = None) -> str:
    if task_name and task_name in TASK_PRIMARY_METRIC:
        return TASK_PRIMARY_METRIC[task_name]
    return CATEGORY_PRIMARY_METRIC.get(task_type, "main_score")


def apply_eval_config(tasks: Iterable[Any]) -> list[Any]:
    """Apply the classification probe protocol to every task.

    Full-train default: class-weighted logistic regression, one draw
    (every draw would see the same rows). With ``ROMTEB_K_SHOT`` set,
    fall back to MTEB's default probe with ``N_EXPERIMENTS`` draws so
    the sample std stays meaningful for a learning-curve report.
    """
    override = _k_shot_override()
    out = []
    for task in tasks:
        name = getattr(getattr(task, "metadata", None), "name", None)
        if name and hasattr(task, "samples_per_label"):
            task.samples_per_label = classification_shots(name)
            if override is None:
                task.n_experiments = 1
                task.evaluator_model = make_probe()
            else:
                task.n_experiments = N_EXPERIMENTS
                task.evaluator_model = LogisticRegression(max_iter=100)
        out.append(task)
    return out