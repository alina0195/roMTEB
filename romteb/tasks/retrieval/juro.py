"""JuRo legal-exam MCQ as retrieval (per-question option pool)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-juro-legal-reranking"


class JuRoLegalExamRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JuRoLegalExamRetrieval",
        description=(
            "Romanian legal-exam MCQ, evaluated as retrieval with a "
            "per-question candidate pool (top_ranked = that question's "
            "options only). Rank the gold option above the distractors. "
            "Primary metric: accuracy@1 (Recall@1); map_at_1000 kept for "
            "MTEB compatibility. Structurally this is the reranking "
            "protocol, published under the Retrieval type per the Sep "
            "2026 taxonomy consolidation."
        ),
        reference="https://github.com/craciuncg/GRAF/tree/main/JuRo",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2025-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt={"query": "Given a Romanian legal exam question, rank the correct answer option first"},
    )
