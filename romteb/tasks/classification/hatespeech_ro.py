"""HateSpeech-RO classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision
from romteb.eval_config import classification_shots


_HF_PATH = "alina0195/romteb-hatespeech-ro"


class HateSpeechROClassification(AbsTaskClassification):
    samples_per_label = classification_shots("HateSpeechROClassification")
    metadata = TaskMetadata(
        name="HateSpeechROClassification",
        description=(
            "Binary hate-speech classification on Romanian social media "
            "comments about the 2018 constitutional referendum."
        ),
        reference="https://github.com/andra-pumnea/hate-speech-ro",
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
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@inproceedings{rohatespeech, title={Romanian Hate Speech}, year={2021}}",
    )
