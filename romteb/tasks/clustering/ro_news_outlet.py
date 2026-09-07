"""RORetrieval native clustering: group articles by news outlet."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClustering

from romteb.data_prep._common import read_revision

_HF_PATH = "alina0195/romteb-roretrieval-outlet-clustering"


class RoNewsOutletClusteringP2P(AbsTaskClustering):
    max_fraction_of_documents_to_embed = 1.0

    metadata = TaskMetadata(
        name="RoNewsOutletClusteringP2P",
        description=(
            "Cluster Romanian articles from RORetrieval by publishing outlet "
            "(adevarul, digi24, protv, mediafax, libertatea, zf, evz, "
            "cotidianul, …). Native Romanian clustering; same source corpus "
            "as RoNewsTypeClusteringP2P (collapsed once in type-macro)."
        ),
        reference="https://github.com/readerbench/RORetrieval",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="v_measure",
        date=("2010-01-01", "2024-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Identify the Romanian news outlet that published this article",
    )
