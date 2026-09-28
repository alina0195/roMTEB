"""Romanian retrieval tasks (BEIR-style).

Open-corpus IR: RoD-TAL, MQA CQA.

MCQ retrieval (per-question option pool, structurally reranking but grouped
under Retrieval per the Sep 2026 taxonomy consolidation): JuRo, WWTBM,
RoMedQA_v2, GRILE.

The full-option-bank MCQ retrieval variants were removed in Sep 2026 after
the discrimination audit (`docs/task_audit.md`) — the option-bank corpus
had duplicate surface forms that collapsed p90 nDCG@10.

MS MARCO RO is held out for training — not a benchmark task.
"""

from romteb.tasks.retrieval.grile import GrileGrammarRetrieval
from romteb.tasks.retrieval.juro import JuRoLegalExamRetrieval
from romteb.tasks.retrieval.mqa_ro import MQARoCQARetrieval
from romteb.tasks.retrieval.rod_tal_retrieval import RoDTALLawsRetrieval
from romteb.tasks.retrieval.romedqa import RoMedQAv2Retrieval
from romteb.tasks.retrieval.wwtbm import WWTBMRoQARetrieval

__all__ = [
    "RoDTALLawsRetrieval",
    "MQARoCQARetrieval",
    "JuRoLegalExamRetrieval",
    "WWTBMRoQARetrieval",
    "RoMedQAv2Retrieval",
    "GrileGrammarRetrieval",
]
