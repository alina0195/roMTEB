"""RoABSA aspect-level sentiment classification."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision
from romteb.eval_config import classification_shots


_HF_PATH = "alina0195/romteb-roabsa"


class RoABSAClassification(AbsTaskClassification):
    samples_per_label = classification_shots("RoABSAClassification")
    metadata = TaskMetadata(
        name="RoABSAClassification",
        description=(
            "Aspect-level sentiment on Romanian product reviews (RoABSA). "
            "Each example concatenates the aspect and the review "
            "('Entitate: {aspect}\\n{body}'); a frozen embedder plus "
            "logistic regression predicts polarity towards that entity."
        ),
        reference="https://huggingface.co/datasets/upb-nlp/RoABSA",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2024-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@inproceedings{roabsa, title={RoABSA}, year={2024}}",
    )
