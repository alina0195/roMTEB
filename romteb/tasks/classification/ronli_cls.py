"""RoNLI 4-class classification with a SentEval-style pair probe.

Same Hub dataset as RoNLIPairClassification (alina0195/romteb-ronli), but all
four upstream classes are kept:

    0 Contrastive, 1 Entailment, 2 Consequence, 3 Unrelated

MTEB classification embeds one text per row. An NLI label is a property of the
*pair*, so each sentence is embedded separately and the probe sees the
InferSent / SentEval feature vector

    [u, v, |u - v|, u * v]        (4 x embedding dim, "4-part")

The 3-part Sentence-BERT variant [u, v, |u - v|] is available for the
appendix ablation: set the environment variable ``ROMTEB_RONLI_PROBE=3part``
before the run (default ``4part``).

The probe (``evaluator_model``), the train budget (``samples_per_label``) and
``n_experiments`` still come from romteb.eval_config, so the full-train
protocol and the k8 sidecar apply unchanged.

Train = upstream train.json (55,102 pairs, distant supervision, labels from
discourse markers that were stripped from the text). Test = upstream
validation + test, deduplicated (5,898 pairs). Both splits are heavily
imbalanced (Contrastive + Entailment are ~5% of test), so the main score is
macro-F1.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from mteb import TaskMetadata
from mteb._create_dataloaders import create_dataloader
from mteb.abstasks import AbsTaskClassification

from romteb.data_prep._common import read_revision


_HF_PATH = "alina0195/romteb-ronli"


def _to_numpy(emb: Any) -> np.ndarray:
    if hasattr(emb, "detach"):  # torch tensor
        emb = emb.detach().float().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)


_PROBE_MODE_ENV = "ROMTEB_RONLI_PROBE"


def _probe_mode() -> str:
    mode = os.environ.get(_PROBE_MODE_ENV, "4part").strip().lower()
    if mode not in {"3part", "4part"}:
        raise ValueError(
            f"{_PROBE_MODE_ENV} must be '3part' or '4part', got {mode!r}"
        )
    return mode


def pair_features(u: np.ndarray, v: np.ndarray, *, mode: str | None = None) -> np.ndarray:
    """SentEval-style pair features. ``mode`` overrides the env-var switch."""
    if (mode or _probe_mode()) == "3part":
        return np.concatenate([u, v, np.abs(u - v)], axis=1)
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=1)


class RoNLIClassification(AbsTaskClassification):
    # Both columns must survive the column selection in AbsTaskClassification.evaluate.
    input_column_name = ["sentence1", "sentence2"]
    label_column_name = "label"

    metadata = TaskMetadata(
        name="RoNLIClassification",
        description=(
            "Romanian NLI (RoNLI, Poesina et al., ACL 2024), all four classes: "
            "Contrastive, Entailment, Consequence, Unrelated. Premise and "
            "hypothesis are embedded separately; a logistic-regression probe is "
            "trained on [u, v, |u-v|, u*v]. Train is the distantly supervised "
            "RONLI train split; test is the manually annotated validation + "
            "test files, deduplicated. Main score: macro-F1."
        ),
        reference="https://github.com/Eduard6421/RONLI",
        dataset={"path": _HF_PATH, "revision": read_revision(_HF_PATH)},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="f1",
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

    def _get_content_columns(self) -> dict[str, str]:
        # Used by descriptive statistics: two text columns, like PairClassification.
        return {"sentence1": "text", "sentence2": "text"}

    def dataset_transform(self, num_proc: int | None = None) -> None:
        keep = {"sentence1", "sentence2", "label"}
        for split in self.dataset:
            ds = self.dataset[split]
            self.dataset[split] = ds.remove_columns(
                [c for c in ds.column_names if c not in keep]
            )

    # ------------------------------------------------------------------
    # Evaluation: same loop as AbsTaskClassification._evaluate_subset, but the
    # "embedding" of a row is the pair feature vector.
    # ------------------------------------------------------------------

    def _encode_column(self, model, ds, column, *, hf_split, hf_subset, encode_kwargs, num_proc):
        dataloader = create_dataloader(
            ds,
            task_metadata=self.metadata,
            input_column=column,
            num_proc=num_proc,
            **encode_kwargs,
        )
        return _to_numpy(
            model.encode(
                dataloader,
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                **encode_kwargs,
            )
        )

    def _pair_embed(self, model, ds, **kw) -> np.ndarray:
        u = self._encode_column(model, ds, "sentence1", **kw)
        v = self._encode_column(model, ds, "sentence2", **kw)
        return pair_features(u, v)

    def _evaluate_subset(
        self,
        model,
        data_split,
        *,
        encode_kwargs,
        hf_split: str,
        hf_subset: str,
        prediction_folder=None,
        num_proc: int | None = None,
        timer,
        **kwargs: Any,
    ):
        train_split = data_split[self.train_split]
        eval_split = data_split[hf_split]

        # Which train rows each experiment uses (all rows under the full-train protocol).
        sim_idxs = None
        all_selected: list[list[int]] = []
        for i in range(self.n_experiments):
            _, sim_idxs, selected = self._undersample_data(train_split, i, sim_idxs)
            all_selected.append(selected)
        union = sorted(set().union(*all_selected))
        pos = {orig: p for p, orig in enumerate(union)}

        with timer("Encoding training pairs", split=hf_split, subset=hf_subset):
            train_feats = self._pair_embed(
                model, train_split.select(union),
                hf_split=self.train_split, hf_subset=hf_subset,
                encode_kwargs=encode_kwargs, num_proc=num_proc,
            )
        with timer("Encoding test pairs", split=hf_split, subset=hf_subset):
            test_feats = self._pair_embed(
                model, eval_split,
                hf_split=hf_split, hf_subset=hf_subset,
                encode_kwargs=encode_kwargs, num_proc=num_proc,
            )

        scores, all_predictions = [], []
        with timer("Scoring", split=hf_split, subset=hf_subset):
            for i in range(self.n_experiments):
                rows = [pos[j] for j in all_selected[i]]
                scores_exp, predictions = self._run_experiment(
                    train_split.select(all_selected[i]),
                    eval_split,
                    train_feats[rows],
                    test_feats,
                    timer=timer,
                )
                scores.append(scores_exp)
                if prediction_folder:
                    all_predictions.append(predictions)

        if prediction_folder:
            self._save_task_predictions(
                all_predictions, model, prediction_folder,
                hf_subset=hf_subset, hf_split=hf_split,
            )
        return self._calculate_avg_scores(scores)