"""RoMath domain classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision
from romteb.eval_config import classification_shots


_HF_PATH = "alina0195/romteb-romath-domain"


class RoMathDomainClassification(AbsTaskClassification):
    samples_per_label = classification_shots("RoMathDomainClassification")
    metadata = TaskMetadata(
        name="RoMathDomainClassification",
        description=(
            "Mathematical domain classification from RoMath. Each example is a "
            "Romanian math problem and the label is its mathematical domain "
            "(e.g. algebra, geometry, analysis). Tests whether sentence "
            "encoders can discriminate topical areas of mathematical text."
        ),
        reference="https://huggingface.co/datasets/cosmadrian/romath",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-12-31"),
        domains=["Academic", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=(
            "@misc{cosma2024romath,\n"
            "  title={RoMath: A Mathematical Reasoning Benchmark in Romanian},\n"
            "  author={Adrian Cosma and Ana-Maria Bucur and Emilian Radoi},\n"
            "  year={2024},\n"
            "  eprint={2409.11074},\n"
            "  archivePrefix={arXiv},\n"
            "  primaryClass={cs.CL},\n"
            "  url={https://arxiv.org/abs/2409.11074}\n"
            "}"
        ),
    )
