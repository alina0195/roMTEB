"""RETIRED from RoMTEB — superseded by RoMedQAv2Reranking / RoMedQAv2Retrieval.

RoMedQA_v2 medical multiple-choice pair-classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskPairClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-romedqa-v2-pair-classification"


class RoMedQAv2PairClassification(AbsTaskPairClassification):
    label_column_name = "label"

    metadata = TaskMetadata(
        name="RoMedQAv2PairClassification",
        description=(
            "Romanian medical multiple-choice QA from RoMedQA_v2, evaluated "
            "as pair classification. Each question is paired with every "
            "answer option; cosine average precision measures whether "
            "correct options score above distractors."
        ),
        reference="https://huggingface.co/datasets/craciuncg/RoMedQA_v2",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="cosine_ap",
        date=("2024-01-01", "2025-12-31"),
        domains=["Medical", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
