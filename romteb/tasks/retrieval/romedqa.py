"""RoMedQA_v2 MCQ as retrieval (per-question option pool)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-romedqa-v2-reranking"


class RoMedQAv2Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RoMedQAv2Retrieval",
        description=(
            "Romanian medical MCQ (RoMedQA_v2), evaluated as retrieval "
            "with a per-question candidate pool (top_ranked = that "
            "question's numbered options). 1–5 gold options per question, "
            "so the primary metric is map_at_1000. Structurally the same "
            "as the reranking protocol; grouped under Retrieval per the "
            "Sep 2026 taxonomy consolidation."
        ),
        reference="https://huggingface.co/datasets/craciuncg/RoMedQA_v2",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="map_at_1000",
        date=("2024-01-01", "2025-12-31"),
        domains=["Medical", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt={
            "query": "Given a Romanian medical exam question, rank the correct answer option first"
        },
    )
