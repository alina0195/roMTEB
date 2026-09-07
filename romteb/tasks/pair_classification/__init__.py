"""Romanian pair-classification tasks.

MCQ datasets (JuRo, WWTBM, RoMedQA) moved to Retrieval + Reranking.
RoNLI stays here: cosine AP on binarized entailment vs contradiction, no LR.
"""

from romteb.tasks.pair_classification.ronli import RoNLIPairClassification

__all__ = ["RoNLIPairClassification"]
