"""RoD-TAL legal information retrieval task (BEIR format)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-rod-tal-retrieval"


class RoDTALLawsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RoDTALLawsRetrieval",
        description=(
            "Legal information retrieval from RoD-TAL. Queries are Romanian "
            "driving-license exam questions; the corpus is composed of "
            "article-level Romanian traffic-law passages. The task tests "
            "whether an encoder can retrieve the relevant legal articles "
            "for a given exam question."
        ),
        reference="https://huggingface.co/datasets/GRAI-UNSTPB/RoD-TAL",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=(
            "@misc{man2025rodtalbenchmarkansweringquestions,\n"
            "  title={RoD-TAL: A Benchmark for Answering Questions in Romanian "
            "Driving License Exams},\n"
            "  author={Andrei Vlad Man and Răzvan-Alexandru Smădu and "
            "Cristian-George Craciun and Dumitru-Clementin Cercel and "
            "Florin Pop and Mihaela-Claudia Cercel},\n"
            "  year={2025},\n"
            "  eprint={2507.19666},\n"
            "  archivePrefix={arXiv},\n"
            "  primaryClass={cs.CL},\n"
            "  url={https://arxiv.org/abs/2507.19666}\n"
            "}"
        ),
    )
