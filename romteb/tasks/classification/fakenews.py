"""Fakenews-RO classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-fakenews-ro"


class FakenewsRoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FakenewsRoClassification",
        description="Binary fake-vs-real classification of Romanian news articles.",
        reference="https://github.com/cipriantruica/Romanian-Fake-News",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2022-12-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@article{truica2022fakenews, title={Romanian Fake News Detection}, year={2022}}",
    )
