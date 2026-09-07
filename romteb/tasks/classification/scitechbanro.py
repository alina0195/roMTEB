"""SciTechBanRO domain/clickbait classification from ClickbaitSciTechRO xlsx."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision
from romteb.eval_config import classification_shots


_HF_PATH = "alina0195/romteb-scitechbanro"


class SciTechBanROClassification(AbsTaskClassification):
    samples_per_label = classification_shots("SciTechBanROClassification")
    metadata = TaskMetadata(
        name="SciTechBanROClassification",
        description=(
            "Classification of Romanian science/technology news from "
            "ClickbaitSciTechRO (xlsx). Frozen embedder + logistic regression."
        ),
        reference="https://github.com/ralucaginga/ClickbaitSciTechRO",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2022-12-31"),
        domains=["News", "Web", "Academic", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@inproceedings{scitechbanro, title={SciTechBanRO}, year={2022}}",
    )
