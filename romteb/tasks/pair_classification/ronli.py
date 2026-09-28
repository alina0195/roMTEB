"""RoNLI pair-classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskPairClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-ronli"

# Upstream RONLI word_to_class: Contrastive 0, Entailment 1, Consequence 2, Unrelated 3.
CONTRASTIVE, ENTAILMENT, CONSEQUENCE, UNRELATED = 0, 1, 2, 3


class RoNLIPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="RoNLIPairClassification",
        description=(
            "Romanian NLI (RoNLI, Poesina et al., ACL 2024): Entailment vs "
            "Contrastive (contradiction), following the MTEB XNLI convention of "
            "dropping the other classes. Consequence and Unrelated pairs "
            "are excluded. Test is the union of the manually annotated RONLI "
            "validation and test files, deduplicated, with no rebalancing "
            "(162 Entailment / 141 Contrastive). See "
            "romteb/data_prep/prep_ronli.py."
        ),
        reference="https://github.com/Eduard6421/RONLI",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="max_ap",
        date=("2023-01-01", "2024-06-30"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=(
            "@inproceedings{poesina2024ronli,\n"
            "  title={A Novel Cartography-Based Curriculum Learning Method Applied on "
            "RoNLI: The First Romanian Natural Language Inference Corpus},\n"
            "  author={Poesina, Eduard Gabriel and Caragea, Cornelia and Ionescu, Radu Tudor},\n"
            "  booktitle={Proceedings of ACL 2024},\n"
            "  year={2024}\n"
            "}"
        ),
    )

    def dataset_transform(self, num_proc: int | None = None) -> None:
        # MTEB PairClassification is binary: Entailment -> 1, Contrastive -> 0.
        for split in self.metadata.eval_splits:
            ds = self.dataset[split].filter(
                lambda x: x["label"] in (ENTAILMENT, CONTRASTIVE)
            )
            ds = ds.map(lambda x: {"labels": 1 if x["label"] == ENTAILMENT else 0})
            keep = {"sentence1", "sentence2", "labels"}
            self.dataset[split] = ds.remove_columns(
                [c for c in ds.column_names if c not in keep]
            )