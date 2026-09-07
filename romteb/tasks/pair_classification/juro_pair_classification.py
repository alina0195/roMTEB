"""RETIRED from RoMTEB — superseded by JuRoLegalExamReranking / JuRoLegalExamRetrieval.

JuRo legal multiple-choice pair-classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskPairClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-juro-legal-pair-classification"


class JuRoLegalExamPairClassification(AbsTaskPairClassification):
    label_column_name = "label"

    metadata = TaskMetadata(
        name="JuRoLegalExamPairClassification",
        description=(
            "Romanian legal-exam multiple-choice QA from JuRo, evaluated as "
            "pair classification. Each question is paired with every answer "
            "option; cosine average precision measures whether correct options "
            "score above distractors."
        ),
        reference="https://github.com/craciuncg/GRAF/tree/main/JuRo",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="cosine_ap",
        date=("2019-01-01", "2025-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
