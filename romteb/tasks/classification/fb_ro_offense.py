"""FB-RO-Offense classification task (license-restricted; may be dropped)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-fb-ro-offense"


class FBRoOffenseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FBRoOffenseClassification",
        description=(
            "Offense classification on Romanian Facebook comments. "
            "Drop this task if the source dataset's redistribution rights "
            "cannot be confirmed."
        ),
        reference="https://huggingface.co/datasets/readerbench/fb-ro-offense",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2024-01-01"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@inproceedings{fbrooffense, title={FB-RO-Offense}, year={2022}}",
    )
