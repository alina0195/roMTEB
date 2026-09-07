"""RoMTEB evaluation knobs shared by task classes and the aggregator.

Classification few-shot
    MTEB default is 8 train examples per class × 10 random draws.
    We raise the budget on harder tasks so the logistic probe sees more
    of the label / aspect space. ``n_experiments`` stays 10 everywhere.

Reporting domain / dataset
    One primary application domain per task (not the generic ``Written``
    tag). ``TASK_DATASET`` maps tasks to a source corpus so domain and
    type means do not double-count MASSIVE. Official MCQ protocol is
    reranking; full-bank Retrieval variants are diagnostic only
    (``MCQ_RETRIEVAL_TASKS``). See ``docs/evaluation_protocol.md`` §9.
"""

from __future__ import annotations

from typing import Any, Iterable

# Models skipped from Borda / plots. Nemotron retrieval is still the
# mis-prompted v1 run (prompt-rerun stopped after bge-multilingual-gemma2).
LEADERBOARD_EXCLUDE: set[str] = {
    "nvidia/llama-embed-nemotron-8b",
}

# Keep the MTEB default unless a task is listed below.
DEFAULT_SAMPLES_PER_LABEL = 8
N_EXPERIMENTS = 10

# Rationale lives in docs/evaluation_protocol.md.
CLASSIFICATION_SHOTS: dict[str, int] = {
    # Coarse document labels (binary / few-class).
    "HateSpeechROClassification": 8,
    "SaRoCoClassification": 8,
    "RomanianReviewsSentiment.v2": 8,
    "RomanianSentimentClassification.v2": 8,
    "SIB200Classification": 8,
    # MASSIVE already has ~60 intents / ~18 scenarios; 8/class is a large pool.
    "MassiveIntentClassification": 8,
    "MassiveScenarioClassification": 8,
    # Fine-grained multi-class.
    "RoOffenseClassification": 16,          # 5 offense classes
    "REDv2EmotionClassification": 16,       # 7 emotions (projected from multi-label)
    "RoMathDomainClassification": 16,       # mathematical domains
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
    "JuRoLegalExamReranking": "Legal",
    "RoDTALLawsRetrieval": "Legal",
    # Medical
    "RoMedQAv2Reranking": "Medical",
    # Academic split: grammar / math / science, plus SIB-200 as general topic.
    "GrileGrammarReranking": "Grammar",
    "RoMathDomainClassification": "Math",
    "SciTechBanROClassification": "Science",
    "SIB200Classification": "Academic",
    "SIB200ClusteringS2S": "Academic",
    "RoNewsOutletClusteringP2P": "News",
    "RoNewsTypeClusteringP2P": "News",
    # Reviews / sentiment
    "RoABSAClassification": "Reviews",
    "RomanianReviewsSentiment.v2": "Reviews",
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
    "WWTBMRoQAReranking": "Culture",
    "WikipediaRetrievalMultilingual": "Encyclopaedic",
    "XQuADRetrieval": "Encyclopaedic",
    # Open-domain web IR
    "WebFAQRetrieval": "Web",
    "MQARoCQARetrieval": "Web",
    "RoNLIPairClassification": "News",
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
OVERALL_EXCLUDE_TASKS: frozenset[str] = frozenset({"JuRoLegalExamReranking"})

# Source dataset for hierarchical aggregation. Two tasks from the same source
# (MASSIVE intent+scenario) must not be counted twice inside one
# (domain, task-type) mean.
TASK_DATASET: dict[str, str] = {
    "HateSpeechROClassification": "HateSpeech-RO",
    "SaRoCoClassification": "SaRoCo",
    "RomanianReviewsSentiment.v2": "RomanianReviews",
    "RomanianSentimentClassification.v2": "RomanianSentiment",
    "SIB200Classification": "SIB-200",
    "SIB200ClusteringS2S": "SIB-200",
    "RoNewsOutletClusteringP2P": "RORetrieval",
    "RoNewsTypeClusteringP2P": "RORetrieval",
    "MassiveIntentClassification": "MASSIVE",
    "MassiveScenarioClassification": "MASSIVE",
    "RoOffenseClassification": "RoOffense",
    "REDv2EmotionClassification": "REDv2",
    "RoMathDomainClassification": "RoMath",
    "SciTechBanROClassification": "SciTechBanRO",
    "HistNERoMentionClassification": "HistNERo",
    "RoABSAClassification": "RoABSA",
    "RoNLIPairClassification": "RoNLI",
    "RoSTS": "RoSTS",
    "WebFAQRetrieval": "WebFAQ",
    "WikipediaRetrievalMultilingual": "Wikipedia",
    "XQuADRetrieval": "XQuAD",
    "RoDTALLawsRetrieval": "RoD-TAL",
    "MQARoCQARetrieval": "MQA-CQA",
    "JuRoLegalExamReranking": "JuRo",
    "WWTBMRoQAReranking": "WWTBM",
    "RoMedQAv2Reranking": "RoMedQA",
    "GrileGrammarReranking": "GRILE",
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
    "JuRoLegalExamReranking": "accuracy",
    "WWTBMRoQAReranking": "accuracy",
    "GrileGrammarReranking": "accuracy",
    "RoMedQAv2Reranking": "map_at_1000",
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
    "JuRoLegalExamReranking": {"k": 3, "accuracy": 0.381, "map_at_1000": 0.639},
    "WWTBMRoQAReranking": {"k": 4, "accuracy": 0.249, "map_at_1000": 0.520},
    "GrileGrammarReranking": {"k": 4, "accuracy": 0.260, "map_at_1000": 0.531},
    "RoMedQAv2Reranking": {"k": 5, "accuracy": 0.342, "map_at_1000": 0.553},
}

# Classification tasks that also print macro-F1 next to accuracy.
REPORT_MACRO_F1: frozenset[str] = frozenset({"RoABSAClassification"})


def classification_shots(task_name: str) -> int:
    return CLASSIFICATION_SHOTS.get(task_name, DEFAULT_SAMPLES_PER_LABEL)


def reporting_domain(task_name: str) -> str | None:
    return TASK_REPORTING_DOMAIN.get(task_name)


def dataset_of(task_name: str) -> str:
    return TASK_DATASET.get(task_name, task_name)


def metric_of(task_type: str, task_name: str | None = None) -> str:
    if task_name and task_name in TASK_PRIMARY_METRIC:
        return TASK_PRIMARY_METRIC[task_name]
    return CATEGORY_PRIMARY_METRIC.get(task_type, "main_score")


def apply_eval_config(tasks: Iterable[Any]) -> list[Any]:
    """Set ``samples_per_label`` / ``n_experiments`` on loaded task objects."""
    out = []
    for task in tasks:
        name = getattr(getattr(task, "metadata", None), "name", None)
        if name and hasattr(task, "samples_per_label"):
            task.samples_per_label = classification_shots(name)
        if name and hasattr(task, "n_experiments"):
            task.n_experiments = N_EXPERIMENTS
        out.append(task)
    return out
