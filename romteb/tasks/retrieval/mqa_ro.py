"""CLiPS MQA Romanian CQA retrieval (community answers)."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-mqa-ro-cqa-retrieval"


class MQARoCQARetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MQARoCQARetrieval",
        description=(
            "Romanian community question answering from CLiPS MQA "
            "(clips/mqa, ro-cqa-question). Queries are CQA questions; the "
            "corpus is filtered answer texts; qrels mark accepted answers. "
            "FAQ subset is omitted (overlaps WebFAQRetrieval). Frozen "
            "bi-encoder: index answers, search with the question, nDCG@10."
        ),
        reference="https://huggingface.co/datasets/clips/mqa",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2022-12-31"),
        domains=["Web", "Social", "Written"],
        task_subtypes=["Question answering"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=(
            "@inproceedings{de-bruyn-etal-2021-mfaq,\n"
            "  title={{MFAQ}: a Multilingual {FAQ} Dataset},\n"
            "  author={De Bruyn, Maxime and Lotfi, Ehsan and "
            "Buhmann, Jeska and Daelemans, Walter},\n"
            "  booktitle={Proceedings of the 3rd Workshop on Machine "
            "Reading for Question Answering},\n"
            "  year={2021},\n"
            "  pages={1--13}\n"
            "}"
        ),
        prompt={
            "query": "Given a Romanian community question, retrieve answers that address the question"
        },
    )
