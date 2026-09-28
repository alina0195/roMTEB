"""WWTBM-Ro MCQ as retrieval (per-question option pool)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-wwtbm-ro"


class WWTBMRoQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WWTBMRoQARetrieval",
        description=(
            "Romanian 'Who Wants to Be a Millionaire?' MCQ, evaluated as "
            "retrieval with a per-question candidate pool (top_ranked = "
            "the four on-screen options). Rank the gold answer first. "
            "Primary metric: accuracy@1; map_at_1000 kept for MTEB "
            "compatibility. Structurally identical to the reranking "
            "protocol; grouped under Retrieval per the Sep 2026 taxonomy "
            "consolidation."
        ),
        reference="https://arxiv.org/abs/2506.05991",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2011-01-01", "2019-12-31"),
        domains=["Spoken", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=(
            "@misc{ganea2025wwtbm,\n"
            "  title={A Culturally-Rich Romanian NLP Dataset from "
            "\"Who Wants to Be a Millionaire?\" Videos},\n"
            "  author={Ganea, Alexandru-Gabriel and others},\n"
            "  year={2025},\n"
            "  eprint={2506.05991}\n"
            "}"
        ),
        prompt={"query": "Given a Romanian quiz question, rank the correct answer first"},
    )
