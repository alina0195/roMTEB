"""HistNERo mention-type classification (gold span + sentence context)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision
from romteb.eval_config import classification_shots


_HF_PATH = "alina0195/romteb-histnero-mentions"


class HistNERoMentionClassification(AbsTaskClassification):
    samples_per_label = classification_shots("HistNERoMentionClassification")
    metadata = TaskMetadata(
        name="HistNERoMentionClassification",
        description=(
            "Historical Romanian mention typing derived from HistNERo. "
            "Each gold NER span is encoded as 'Mențiune: {span}\\nContext: "
            "{sentence}'; a logistic-regression probe predicts "
            "PERS/ORG/LOC/PROD/DATE. Not token-level NER."
        ),
        reference="https://huggingface.co/datasets/avramandrei/histnero",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("1817-01-01", "1990-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=(
            "@article{avram2024histnero,\n"
            "  title={HistNERo: Historical Named Entity Recognition for the Romanian Language},\n"
            "  author={Avram, Andrei-Marius and others},\n"
            "  journal={arXiv preprint arXiv:2405.00155},\n"
            "  year={2024}\n"
            "}"
        ),
    )
