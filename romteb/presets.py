"""Named task subsets for platform / CI callers.

``smoke``  one cheap STS task (minutes).
``core``   Classification + PairClassification + STS (no IR).
``full``   official v1 protocol: everything except clustering.

BitextMining is included in ``full`` (reported separately, not in Overall).
"""

from __future__ import annotations

from typing import Any, Iterable

PROTOCOL_NAME = "romteb-v1"

SMOKE_TASKS: tuple[str, ...] = ("RoSTS",)
CORE_TYPES: frozenset[str] = frozenset(
    {"Classification", "PairClassification", "STS"}
)
FULL_EXCLUDE_TYPES: frozenset[str] = frozenset({"Clustering"})
PRESET_NAMES: tuple[str, ...] = ("smoke", "core", "full")


def _task_name(task: Any) -> str:
    return getattr(getattr(task, "metadata", None), "name", "") or ""


def _task_type(task: Any) -> str:
    return getattr(getattr(task, "metadata", None), "type", "") or ""


def _infer_type(name: str) -> str:
    if "PairClassification" in name or name.endswith("PairClassification"):
        return "PairClassification"
    if name.endswith("Classification") or name.endswith(".v2"):
        return "Classification"
    if "Retrieval" in name:
        return "Retrieval"
    if "Reranking" in name:
        return "Reranking"
    if "BitextMining" in name or name == "Tatoeba":
        return "BitextMining"
    if "Clustering" in name:
        return "Clustering"
    if "Summarization" in name:
        return "Summarization"
    if "STS" in name or name == "RonSTS":
        return "STS"
    return ""


class _CatalogItem:
    def __init__(self, name: str, type_name: str):
        self.metadata = type("M", (), {"name": name, "type": type_name})()


def static_catalog() -> list[_CatalogItem]:
    """Protocol roster from eval_config (no mteb import)."""
    from romteb.eval_config import TASK_DATASET

    return [_CatalogItem(name, _infer_type(name)) for name in TASK_DATASET]


def _names_for_preset(preset: str, tasks: Iterable[Any]) -> list[str]:
    items = [t for t in tasks if _task_name(t)]
    if preset == "smoke":
        available = {_task_name(t) for t in items}
        missing = [n for n in SMOKE_TASKS if n not in available]
        if missing:
            raise SystemExit(f"[romteb] smoke preset missing tasks: {missing}")
        return list(SMOKE_TASKS)
    if preset == "core":
        return [_task_name(t) for t in items if _task_type(t) in CORE_TYPES]
    if preset == "full":
        return [_task_name(t) for t in items if _task_type(t) not in FULL_EXCLUDE_TYPES]
    raise SystemExit(f"[romteb] unknown preset {preset!r}; use {list(PRESET_NAMES)}")


def resolve_task_names(
    *,
    preset: str | None,
    tasks: list[str] | None,
    catalog: Iterable[Any],
) -> list[str] | None:
    """Return explicit task names, or None to keep the runner default.

    ``--tasks`` wins over ``--preset``. ``full`` is equivalent to the runner
    default (all tasks except clustering).
    """
    if tasks:
        if preset and preset != "full":
            print(
                f"[romteb] --tasks overrides --preset {preset}",
                flush=True,
            )
        return tasks
    if not preset or preset == "full":
        return None
    return _names_for_preset(preset, catalog)


def describe_presets(catalog: Iterable[Any] | None = None) -> dict[str, Any]:
    items = list(catalog) if catalog is not None else static_catalog()
    items = [t for t in items if _task_name(t)]
    by_preset = {name: _names_for_preset(name, items) for name in PRESET_NAMES}
    return {
        "protocol": PROTOCOL_NAME,
        "presets": by_preset,
        "tasks": [
            {"name": _task_name(t), "type": _task_type(t)} for t in items
        ],
    }
