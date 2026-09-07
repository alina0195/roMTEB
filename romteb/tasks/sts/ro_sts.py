"""RO-STS — RoMTEB reference task.

This file is the canonical template for every custom RoMTEB task. Field
shapes follow MTEB v2 (see `mteb.abstasks.TaskMetadata`). If MTEB rejects
any field, fix it here first, then `replace_all` the same change across
every other task class.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskSTS

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-ro-sts"


class RoSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="RoSTS",
        description=(
            "Romanian Semantic Textual Similarity. Sentence pairs annotated "
            "with continuous similarity scores in [0, 5]. Derived from the "
            "STSBenchmark, translated to Romanian by Dumitrescu et al."
        ),
        reference="https://huggingface.co/datasets/dumitrescustefan/ro_sts",
        dataset={
            "path": _HF_PATH,
            "revision": read_revision(_HF_PATH),
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="cosine_spearman",
        date=("2020-01-01", "2021-12-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=(
            "@inproceedings{dumitrescu-etal-2021-liro,\n"
            "  title={LiRo: Benchmark and leaderboard for Romanian language tasks},\n"
            "  author={Dumitrescu, Stefan Daniel and others},\n"
            "  booktitle={NeurIPS Datasets and Benchmarks},\n"
            "  year={2021}\n"
            "}"
        ),
    )

    min_score = 0.0
    max_score = 5.0
