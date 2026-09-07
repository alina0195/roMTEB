"""Enumerate every MTEB task that ships with a Romanian subset.

Run this once after building the romteb.sif image:

    ./apptainer-exec-romteb.sh romteb/scripts/inventory_mteb_ron.py

Writes:
  - romteb/tasks/_inventory.md   : markdown table of all romanian tasks
  - romteb/tasks/_inventory.json : machine-readable version
  - stdout                       : same table for quick inspection

The inventory drives Phase 2: any task with `decision=reuse` gets imported
directly in `benchmark.py`; everything else must be reimplemented in
`romteb/tasks/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mteb


RON_KEYS = {"ron", "ron-Latn", "ron_Latn", "ro", "ro-RO", "Romanian"}

# Manual overrides applied after the MTEB scan. Keeps the inventory aligned with
# `benchmark.py` even when a task still exists upstream in mteb.
INVENTORY_OVERRIDES: dict[str, dict[str, str]] = {
    "FloresBitextMining": {
        "decision": "exclude",
        "reason": "only a devtest split in mteb 2.12.x; no usable score",
    },
    "RonSTS": {
        "decision": "custom",
        "reason": "same RO-STS data as custom RoSTS; RonSTS dropped from RoMTEB",
    },
}


def _extract_ron_subsets(task: object) -> list[str]:
    meta = getattr(task, "metadata", None)
    if meta is None:
        return []
    langs = getattr(meta, "eval_langs", None)
    if langs is None:
        return []
    if isinstance(langs, dict):
        return [
            subset
            for subset, lang_list in langs.items()
            if any(_matches_ron(code) for code in lang_list)
            or _matches_ron(subset)
        ]
    if isinstance(langs, list):
        return ["default"] if any(_matches_ron(code) for code in langs) else []
    return []


def _matches_ron(code: str) -> bool:
    if not isinstance(code, str):
        return False
    return any(key.lower() in code.lower() for key in RON_KEYS)


def main() -> None:
    print(f"mteb version: {mteb.__version__}")
    try:
        tasks = mteb.get_tasks(languages=["ron"])
    except TypeError:
        tasks = mteb.get_tasks(languages=["ron-Latn"])
    print(f"Found {len(tasks)} tasks with a Romanian subset.\n")

    rows: list[dict] = []
    for t in tasks:
        meta = t.metadata
        ron_subsets = _extract_ron_subsets(t) or ["default"]
        dataset = getattr(meta, "dataset", None) or {}
        path = dataset.get("path") if isinstance(dataset, dict) else getattr(dataset, "path", "?")
        row = {
            "task_name": meta.name,
            "type": str(meta.type),
            "ron_subsets": ron_subsets,
            "hf_path": path,
            "main_score": getattr(meta, "main_score", "?"),
            "decision": "reuse",
            "reason": "default; review and override if needed",
        }
        override = INVENTORY_OVERRIDES.get(meta.name)
        if override:
            row.update(override)
        rows.append(row)

    rows.sort(key=lambda r: (r["type"], r["task_name"]))

    out_md = Path(__file__).resolve().parent.parent / "romteb" / "tasks" / "_inventory.md"
    out_json = out_md.with_suffix(".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# MTEB Romanian task inventory",
        "",
        f"Generated from `mteb=={mteb.__version__}`. Edit the `decision` and `reason`",
        "columns by hand after this script runs; rerun to refresh the list.",
        "",
        "| task_name | type | ron_subsets | hf_path | main_score | decision | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        subsets = ", ".join(r["ron_subsets"])
        lines.append(
            f"| `{r['task_name']}` | {r['type']} | {subsets} | "
            f"`{r['hf_path']}` | {r['main_score']} | {r['decision']} | {r['reason']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {out_md}\nWrote {out_json}")


if __name__ == "__main__":
    main()
