"""Ro-Offense classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision
from romteb.eval_config import classification_shots


_HF_PATH = "alina0195/romteb-ro-offense"


class RoOffenseClassification(AbsTaskClassification):
    samples_per_label = classification_shots("RoOffenseClassification")
    metadata = TaskMetadata(
        name="RoOffenseClassification",
        description=(
            "5-class offense classification on Romanian news comments "
            "(readerbench/news-ro-offense)."
        ),
        reference="https://huggingface.co/datasets/readerbench/news-ro-offense",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2023-01-01"),
        domains=["News", "Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@inproceedings{ro-offense, title={Ro-Offense}, year={2022}}",
    )
