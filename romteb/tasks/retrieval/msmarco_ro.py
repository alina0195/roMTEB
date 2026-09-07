"""MS MARCO Romanian retrieval — **not registered in RoMTEB**.

Held out for training / train-time eval on `alina0195/ro-msmarco-divided`.
Using the same test split on the leaderboard would contaminate any model
trained on this corpus.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-msmarco-ro-retrieval"


class MSMarcoRoRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMarcoRoRetrieval",
        description=(
            "Romanian translation of MS MARCO passage ranking. Queries are "
            "the unique anchors from alina0195/ro-msmarco-divided split=test; "
            "the corpus is unique test positives/negatives plus eval-split "
            "passages as extra distractors; qrels mark the positive passage. "
            "Frozen bi-encoder: index passages, search with the query, nDCG@10."
        ),
        reference="https://huggingface.co/datasets/alina0195/ro-msmarco-divided",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="ndcg_at_10",
        date=("2016-01-01", "2025-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="msr-la-nc",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=(
            "@article{DBLP:journals/corr/NguyenRSGTMD16,\n"
            "  title={{MS} {MARCO:} A Human Generated MAchine Reading "
            "COmprehension Dataset},\n"
            "  author={Nguyen, Tri and others},\n"
            "  journal={CoRR},\n"
            "  volume={abs/1611.09268},\n"
            "  year={2016}\n"
            "}"
        ),
        prompt={
            "query": "Given a web search query in Romanian, retrieve relevant passages that answer the query"
        },
    )
