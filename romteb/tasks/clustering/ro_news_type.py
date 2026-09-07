"""RORetrieval native clustering: group articles by content type."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClustering

from romteb.data_prep._common import read_revision

_HF_PATH = "alina0195/romteb-roretrieval-type-clustering"


class RoNewsTypeClusteringP2P(AbsTaskClustering):
    max_fraction_of_documents_to_embed = 1.0

    metadata = TaskMetadata(
        name="RoNewsTypeClusteringP2P",
        description=(
            "Cluster Romanian articles from RORetrieval by content type "
            "(news, recipes, stories, …). Native Romanian clustering; same "
            "source corpus as RoNewsOutletClusteringP2P (TASK_DATASET="
            "RORetrieval, counted once in Clustering type-macro)."
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
        prompt="Identify the content type of this Romanian article",
    )
