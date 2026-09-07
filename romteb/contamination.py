"""Contamination is a property of (embedder, task), not of a task alone.

A cell is contaminated when the embedder's published training data overlaps
the evaluation source family. Models with no public training-data statement
are ``unknown`` — reported as such, not dropped from means.

See docs/evaluation_protocol.md §9.
"""

from __future__ import annotations

from typing import Literal

ContaminationStatus = Literal["contaminated", "clean", "unknown"]

# Families of training / eval sources. A (model, task) pair is contaminated
# when the model's training set intersects the task's family.
STS_B = "STS-B"
MSMARCO = "MSMARCO"
MSMARCO_RO = "mMARCO-ro"
WEBFAQ_RO_TRAIN = "WebFAQ-ro-train"
SQUAD_XQUAD = "SQuAD/XQuAD"
NQ = "NQ"
WIKIPEDIA_RETRIEVAL = "Wikipedia-retrieval-train"

TASK_SOURCE_FAMILY: dict[str, frozenset[str]] = {
    "RoSTS": frozenset({STS_B}),
    "WebFAQRetrieval": frozenset({WEBFAQ_RO_TRAIN}),
    "MSMarcoRoRetrieval": frozenset({MSMARCO_RO, MSMARCO}),
    "XQuADRetrieval": frozenset({SQUAD_XQUAD}),
    "WikipediaRetrievalMultilingual": frozenset({WIKIPEDIA_RETRIEVAL}),
}

# ``None`` = training data not published (unknown). Empty frozenset = published
# and no overlap with RoMTEB eval families we track.
MODEL_TRAINING_DATA: dict[str, frozenset[str] | None] = {
    "intfloat/multilingual-e5-large": frozenset({MSMARCO, NQ, SQUAD_XQUAD}),
    "intfloat/multilingual-e5-base": frozenset({MSMARCO, NQ, SQUAD_XQUAD}),
    "intfloat/multilingual-e5-small": frozenset({MSMARCO, NQ, SQUAD_XQUAD}),
    "BlackKakapo/stsb-xlm-r-multilingual-ro": frozenset({STS_B}),
    "BAAI/bge-m3": frozenset({MSMARCO, NQ}),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": frozenset(),
    "sentence-transformers/LaBSE": frozenset(),
    "iliemihai/romanian-sentence-bert-base-uncased-v1": None,
    "ibm-granite/granite-embedding-97m-multilingual-r2": frozenset(
        {NQ, WIKIPEDIA_RETRIEVAL}
    ),
    "Qwen/Qwen3-Embedding-4B": frozenset({MSMARCO, NQ}),
    # Fill in when the training mix is frozen.
    "alina0195/ro-retriever": None,
    "alina0195/robert-retriever": None,
    "romteb/bm25s-ro": frozenset(),
}

CONTAM_MARK = " *"
UNKNOWN_MARK = " ?"


def _canon(name: str) -> str:
    return name.replace("__", "/")


def training_data_of(model_name_or_slug: str) -> frozenset[str] | None:
    """Return published training families, or None if unknown."""
    key = _canon(model_name_or_slug)
    if key in MODEL_TRAINING_DATA:
        return MODEL_TRAINING_DATA[key]
    return None


def contamination_status(model_name_or_slug: str, task_name: str) -> ContaminationStatus:
    """Classify one (embedder, task) cell."""
    family = TASK_SOURCE_FAMILY.get(task_name)
    if not family:
        return "clean"
    trained = training_data_of(model_name_or_slug)
    if trained is None:
        return "unknown"
    if trained & family:
        return "contaminated"
    return "clean"


def contamination_reason(model_name_or_slug: str, task_name: str) -> str | None:
    """Human-readable reason, or None if the cell is clean."""
    status = contamination_status(model_name_or_slug, task_name)
    if status == "clean":
        return None
    family = TASK_SOURCE_FAMILY.get(task_name, frozenset())
    trained = training_data_of(model_name_or_slug)
    if status == "unknown":
        return "training data not published"
    overlap = sorted(trained & family) if trained else []
    return "trained on " + ", ".join(overlap)


def iter_known_contaminated() -> list[tuple[str, str, str]]:
    """(model, task, reason) for every confirmed contaminated pair we can name."""
    out: list[tuple[str, str, str]] = []
    for model in MODEL_TRAINING_DATA:
        for task in TASK_SOURCE_FAMILY:
            if contamination_status(model, task) == "contaminated":
                reason = contamination_reason(model, task) or ""
                out.append((model, task, reason))
    return out
