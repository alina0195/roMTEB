"""MedQARo retrieval task (clinical QA over Romanian discharge notes)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-medqa-ro"


class MedQARoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MedQARoRetrieval",
        description=(
            "Clinical QA retrieval in Romanian from MedQARo. "
            "Queries are medically grounded questions extracted from "
            "oncology discharge notes, and the corpus contains the "
            "corresponding Romanian epicriza texts."
        ),
        reference="https://huggingface.co/datasets/alina0195/romteb-medqa-ro",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="ndcg_at_10",
        date=("2016-01-01", "2024-12-31"),
        domains=["Medical", "Written"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
    )
