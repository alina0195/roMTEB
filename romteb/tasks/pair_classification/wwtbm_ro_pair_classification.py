"""RETIRED from RoMTEB — superseded by WWTBMRoQAReranking / WWTBMRoQARetrieval.

WWTBM-Romanian cultural multiple-choice pair-classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskPairClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-wwtbm-ro-pair-classification"


class WWTBMRoQAPairClassification(AbsTaskPairClassification):
    label_column_name = "label"

    metadata = TaskMetadata(
        name="WWTBMRoQAPairClassification",
        description=(
            "Cultural multiple-choice QA in Romanian from the WWTBM "
            "Romanian edition of 'Who Wants to Be a Millionaire?'. Each "
            "question is paired with its four on-screen answer options; "
            "cosine average precision measures whether the gold option "
            "scores above distractors."
        ),
        reference="https://arxiv.org/abs/2506.05991",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="cosine_ap",
        date=("2011-01-01", "2019-12-31"),
        domains=["Spoken", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=(
            "@misc{ganea2025wwtbm,\n"
            "  title={A Culturally-Rich Romanian NLP Dataset from\n"
            "         \"Who Wants to Be a Millionaire?\" Videos},\n"
            "  author={Ganea, Alexandru-Gabriel and Popovici,\n"
            "          Antonia-Adelina and Dumitran, Adrian-Marius},\n"
            "  year={2025},\n"
            "  eprint={2506.05991},\n"
            "  archivePrefix={arXiv},\n"
            "  primaryClass={cs.CL}\n"
            "}"
        ),
    )
