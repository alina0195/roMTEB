"""RoNLI pair-classification task."""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskPairClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-ronli"


class RoNLIPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="RoNLIPairClassification",
        description=(
            "Romanian Natural Language Inference (entailment vs contradiction). "
            "Neutral pairs are filtered out and labels are binarized following "
            "the MTEB XNLI convention. Derived from Eduard6421/RONLI on GitHub. "
            "Test split is rebalanced by undersampling contradiction so the "
            "post-filter entailment prevalence is ~50% (74 entailment / 74 "
            "contradiction). See romteb/data_prep/prep_ronli.py."
        ),
        reference="https://github.com/Eduard6421/RONLI",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="max_ap",
        date=("2022-01-01", "2024-01-01"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="@misc{ronli, title={RoNLI}, author={Birsan, Eduard et al.}, year={2023}}",
    )

    def dataset_transform(self, num_proc: int | None = None) -> None:
        # MTEB PairClassification is binary (0/1). Match XNLI: drop neutral,
        # map entailment (0) -> 1 and contradiction (2) -> 0.
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split].filter(
                lambda x: x["label"] in [0, 2]  # noqa: PLR6201
            )
            hf_dataset = hf_dataset.map(
                lambda example: {"labels": 0 if example["label"] == 2 else 1}
            )
            self.dataset[split] = hf_dataset.remove_columns(["label"])
