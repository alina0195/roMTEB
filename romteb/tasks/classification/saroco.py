"""SaRoCo Romanian sarcasm classification."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision
from romteb.eval_config import classification_shots


_HF_PATH = "alina0195/romteb-saroco"


class SaRoCoClassification(AbsTaskClassification):
    samples_per_label = classification_shots("SaRoCoClassification")
    metadata = TaskMetadata(
        name="SaRoCoClassification",
        description=(
            "Romanian sarcasm detection (SaRoCo). Each comment/tweet is "
            "encoded frozen and a logistic-regression probe predicts sarcasm."
        ),
        reference="https://github.com/MihaelaGaman/SaRoCo",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2021-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=(
            "@inproceedings{gaman-ionescu-2021-saroco,\n"
            "  title={SaRoCo: Detecting Satire in a Novel {R}omanian Corpus of News Articles},\n"
            "  author={Gaman, Mihaela and Ionescu, Radu Tudor},\n"
            "  year={2021}\n"
            "}"
        ),
    )
