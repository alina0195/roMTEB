"""Romanian retrieval tasks (BEIR-style).

Open-corpus IR: RoD-TAL, MQA CQA.
The four full-option-bank MCQ retrieval variants (Grile / JuRo / WWTBM /
RoMedQA *Retrieval*) were removed in Sep 2026 after the discrimination
audit (`docs/task_audit.md`). The official MCQ protocol lives in
`romteb.tasks.reranking` (`*Reranking`).
MS MARCO RO is held out for training — not a benchmark task.
"""

from romteb.tasks.retrieval.mqa_ro import MQARoCQARetrieval
from romteb.tasks.retrieval.rod_tal_retrieval import RoDTALLawsRetrieval

__all__ = [
    "RoDTALLawsRetrieval",
    "MQARoCQARetrieval",
]
