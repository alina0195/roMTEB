"""GRILE grammar-exam reranking (4 options per question)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-grile-reranking"


class GrileGrammarReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GrileGrammarReranking",
        description=(
            "Romanian grammar MCQ (GRILE) as reranking. For each item, "
            "top_ranked is options a–d; the frozen embedder ranks the gold "
            "letter first. This is the natural exam protocol. "
            "Primary metric: accuracy@1; map_at_1000 kept for compatibility."
        ),
        reference="https://huggingface.co/datasets/alina0195/romteb-grile-reranking",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Reranking",
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
