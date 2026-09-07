"""RoMedQA_v2 reranking (per-question option pool)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-romedqa-v2-reranking"


class RoMedQAv2Reranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RoMedQAv2Reranking",
        description=(
            "Romanian medical MCQ (RoMedQA_v2) as reranking. top_ranked is "
            "the numbered options of that question; rank gold option(s) "
            "first. Multi-relevant (1–5 gold options); primary metric map_at_1000."
        ),
        reference="https://huggingface.co/datasets/craciuncg/RoMedQA_v2",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Reranking",
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
