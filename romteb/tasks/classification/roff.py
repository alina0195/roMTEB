"""ROFF (Romanian Twitter offense) classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-roff"


class ROFFClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ROFFClassification",
        description="Offensive vs. non-offensive classification of Romanian tweets.",
        reference="https://huggingface.co/datasets/readerbench/ROFF",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2024-01-01"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@inproceedings{roff, title={ROFF: Romanian Offensive Tweets}, year={2022}}",
    )
