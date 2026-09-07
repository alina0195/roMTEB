"""REDv2 emotion classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision
from romteb.eval_config import classification_shots


_HF_PATH = "alina0195/romteb-red-v2-emotion"


class REDv2EmotionClassification(AbsTaskClassification):
    samples_per_label = classification_shots("REDv2EmotionClassification")
    metadata = TaskMetadata(
        name="REDv2EmotionClassification",
        description=(
            "Romanian emotion classification based on REDv2 tweets. "
            "The original multi-label annotations are projected to a single "
            "dominant label (unique argmax over annotator vote counts) "
            "to provide a strict single-label benchmark."
        ),
        reference="https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets/tree/main/REDv2",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2022-01-01", "2022-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=(
            "@inproceedings{redv2,\n"
            "  title={RED v2: Enhancing RED Dataset for Multi-Label Emotion Detection},\n"
            "  author={Ciobotaru, Alexandra and Constantinescu, Mihai V. and Dinu, Liviu P. and Dumitrescu, Stefan Daniel},\n"
            "  year={2022},\n"
            "  booktitle={Proceedings of LREC 2022}\n"
            "}"
        ),
    )
