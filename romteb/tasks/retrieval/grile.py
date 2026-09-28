"""GRILE grammar-exam MCQ as retrieval (4 options per question)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-grile-reranking"


class GrileGrammarRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GrileGrammarRetrieval",
        description=(
            "Romanian grammar MCQ (GRILE), evaluated as retrieval with a "
            "per-question candidate pool (top_ranked = options a–d). Rank "
            "the gold letter first. Primary metric: accuracy@1; "
            "map_at_1000 kept for MTEB compatibility. Structurally the "
            "same as the reranking protocol; grouped under Retrieval per "
            "the Sep 2026 taxonomy consolidation."
        ),
        reference="https://huggingface.co/datasets/alina0195/romteb-grile-reranking",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2024-12-31"),
        domains=["Academic", "Written"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt={
            "query": "Given a Romanian grammar exam question, rank the correct answer option first"
        },
    )
