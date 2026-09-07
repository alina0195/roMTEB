"""Single-model ``summary.json`` for platform callers.

Overall Borda is a rank against other models and is **not** computed here.
Type-macro / task-macro follow the same exclusions as ``scripts/aggregate_results.py``
(BitextMining, saturated tasks, overall-excluded tasks, confirmed contamination).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from romteb.bm25_ro import LEXICAL_BASELINE_NAMES
from romteb.contamination import contamination_reason, contamination_status
from romteb.eval_config import (
    CATEGORY_PRIMARY_METRIC,
    OVERALL_EXCLUDE_TASKS,
    REPORT_MACRO_F1,
    SATURATED_TASKS,
    dataset_of,
    metric_of,
    reporting_domain,
)
from romteb import __version__
from romteb.presets import PROTOCOL_NAME
from romteb.scoring import extract_score_stats, stderr

CROSSLINGUAL_CATEGORIES = frozenset({"BitextMining"})
SKIP_RESULT_FILES = {"model_meta.json", "_index.json", "summary.json"}
SKIP_TOP_DIRS = {"predictions", "plots"}

CATEGORY_ORDER = [
    "Classification",
    "PairClassification",
    "STS",
    "Retrieval",
    "Reranking",
    "BitextMining",
    "Clustering",
    "Summarization",
]

# Official type-macro / Overall categories. Clustering is v1-optional;
# BitextMining is a separate cross-lingual section.
TYPE_MACRO_CATEGORIES = (
    "Classification",
    "PairClassification",
    "STS",
    "Retrieval",
    "Reranking",
)

OVERALL_NOTE = (
    "Overall Borda is a rank against the official roster, not a score of an "
    "isolated model. Use scores[] and by_type. type_macro averages "
    "Classification, PairClassification, STS, Retrieval, and Reranking "
    "dataset means after dropping BitextMining, Clustering, saturated tasks "
    f"({', '.join(sorted(SATURATED_TASKS)) or 'none'}), overall-excluded tasks "
    f"({', '.join(sorted(OVERALL_EXCLUDE_TASKS)) or 'none'}), and confirmed "
    "contaminated cells. Never mix metrics inside a mean."
)


def _canon_model(name: str) -> str:
    return name.replace("__", "/")


def _slug(name: str) -> str:
    return name.replace("/", "__")


def _is_lexical(model: str) -> bool:
    return _canon_model(model) in LEXICAL_BASELINE_NAMES or model in {
        n.replace("/", "__") for n in LEXICAL_BASELINE_NAMES
    }


def infer_category(task_name: str, result: dict | None = None) -> str | None:
    if result:
        category = result.get("task_type")
        if category in CATEGORY_PRIMARY_METRIC:
            return category
    if "PairClassification" in task_name or task_name.endswith("PairClassification"):
        return "PairClassification"
    if task_name.endswith("Classification") or task_name.endswith(".v2"):
        return "Classification"
    if "Retrieval" in task_name:
        return "Retrieval"
    if "Reranking" in task_name:
        return "Reranking"
    if "BitextMining" in task_name or task_name == "Tatoeba":
        return "BitextMining"
    if "Clustering" in task_name:
        return "Clustering"
    if "Summarization" in task_name:
        return "Summarization"
    if "STS" in task_name or task_name == "RonSTS":
        return "STS"
    return None


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _walk_model_results(
    results_dir: Path,
    model: str | None,
    *,
    allow_empty: bool = False,
) -> tuple[str, dict[str, dict], dict[str, dict], dict]:
    """Return (model_id, task->json, k8 task->json, model_meta)."""
    wanted = _slug(model) if model else None
    chosen: dict[str, dict[str, tuple[float, dict]]] = defaultdict(dict)
    k8: dict[str, dict[str, tuple[float, dict]]] = defaultdict(dict)
    meta_by_model: dict[str, tuple[float, dict]] = {}

    for path in results_dir.rglob("*.json"):
        if path.name in SKIP_RESULT_FILES:
            if path.name == "model_meta.json":
                rel = path.relative_to(results_dir)
                slug = rel.parts[0] if rel.parts else "unknown"
                payload = _load_json(path)
                if payload:
                    mtime = path.stat().st_mtime
                    prev = meta_by_model.get(slug)
                    if prev is None or mtime >= prev[0]:
                        meta_by_model[slug] = (mtime, payload)
            continue
        rel = path.relative_to(results_dir)
        top = rel.parts[0] if rel.parts else ""
        if top in SKIP_TOP_DIRS or top.startswith("_"):
            continue
        if wanted and top != wanted and _canon_model(top) != _canon_model(wanted):
            continue
        data = _load_json(path)
        if data is None:
            continue
        is_k8 = path.name.endswith(".k8.json")
        stem = path.stem[: -len(".k8")] if is_k8 else path.stem
        task_name = data.get("task_name") or stem
        if is_k8 and isinstance(task_name, str) and task_name.endswith(".k8"):
            task_name = task_name[: -len(".k8")]
        if task_name == "RonSTS":
            task_name = "RoSTS"
        mtime = path.stat().st_mtime
        bucket = k8 if is_k8 else chosen
        prev = bucket[top].get(task_name)
        if prev is None or mtime >= prev[0]:
            bucket[top][task_name] = (mtime, data)

    if not chosen:
        hint = f" for model {model}" if model else ""
        if allow_empty and model:
            return _canon_model(model), {}, {}, {}
        raise SystemExit(f"[romteb] no task JSON under {results_dir}{hint}")

    if wanted:
        slugs = [
            s
            for s in chosen
            if s == wanted or _canon_model(s) == _canon_model(wanted)
        ]
        if not slugs:
            raise SystemExit(
                f"[romteb] no results for {model} under {results_dir}"
            )
        slug = slugs[0]
    elif len(chosen) == 1:
        slug = next(iter(chosen))
    else:
        names = sorted(_canon_model(s) for s in chosen)
        raise SystemExit(
            "[romteb] multiple models under "
            f"{results_dir}: {names}. Pass --model."
        )

    tasks = {name: payload for name, (_mt, payload) in chosen[slug].items()}
    k8_tasks = {
        name: payload for name, (_mt, payload) in k8.get(slug, {}).items()
    }
    meta = meta_by_model.get(slug, (0.0, {}))[1]
    return _canon_model(slug), tasks, k8_tasks, meta


def _max_seq_length(meta: dict) -> int | None:
    for key in ("max_tokens", "max_seq_length", "romteb_max_seq_length"):
        v = meta.get(key)
        if isinstance(v, (int, float)) and v == v:
            return int(v)
    return None


def _majority_metric(metrics: list[str], task_type: str) -> str:
    counts = Counter(metrics)
    best_n = max(counts.values())
    tied = sorted(m for m, n in counts.items() if n == best_n)
    if len(tied) == 1:
        return tied[0]
    primary = CATEGORY_PRIMARY_METRIC.get(task_type)
    if primary in tied:
        return primary
    if task_type == "Reranking" and "accuracy" in tied:
        return "accuracy"
    return tied[0]


def _in_aggregates(task: str, task_type: str, contaminated: bool, lexical: bool) -> bool:
    if lexical or contaminated:
        return False
    if task in SATURATED_TASKS or task in OVERALL_EXCLUDE_TASKS:
        return False
    return True


def _in_type_macro(task: str, task_type: str, contaminated: bool, lexical: bool) -> bool:
    if task_type in CROSSLINGUAL_CATEGORIES:
        return False
    return _in_aggregates(task, task_type, contaminated, lexical)


def _dataset_means_by_type(
    score_rows: list[dict],
    *,
    flag: str = "in_type_macro",
) -> dict[str, list[tuple[str, float]]]:
    """One (metric, mean) per (task_type, dataset); MASSIVE intent+scenario collapse."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in score_rows:
        if not row.get(flag):
            continue
        grouped[(row["task_type"], row["dataset"], row["metric"])].append(
            float(row["value"])
        )
    per_dataset: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for (task_type, dataset, metric), vals in grouped.items():
        per_dataset[(task_type, dataset)].append((metric, mean(vals)))
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (task_type, _dataset), pairs in per_dataset.items():
        chosen = _majority_metric([m for m, _ in pairs], task_type)
        for metric, value in pairs:
            if metric == chosen:
                out[task_type].append((metric, value))
                break
    return out


def build_summary(
    results_dir: str | Path,
    *,
    model: str | None = None,
    failed_tasks: list[str] | None = None,
    preset: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    root = Path(results_dir)
    if not root.exists():
        raise SystemExit(f"[romteb] results dir not found: {root}")

    model_id, by_task, k8_by_task, meta = _walk_model_results(
        root, model, allow_empty=bool(failed_tasks)
    )
    lexical = _is_lexical(model_id)
    failed = list(failed_tasks or [])
    scores: list[dict[str, Any]] = []

    for task_name, result in sorted(by_task.items()):
        task_type = infer_category(task_name, result) or ""
        metric = metric_of(task_type, task_name)
        stats = extract_score_stats(result, metric_key=metric)
        if stats is None:
            continue
        status = contamination_status(model_id, task_name)
        k8_stats = None
        if task_name in k8_by_task:
            k8_stats = extract_score_stats(k8_by_task[task_name], metric_key="accuracy")
        se = stderr(stats)
        row = {
            "task": task_name,
            "dataset": dataset_of(task_name),
            "domain": reporting_domain(task_name),
            "task_type": task_type,
            "metric": metric,
            "value": _round(stats.mean),
            "std": _round(stats.std),
            "stderr": _round(se),
            "n": stats.n,
            "n_subsets": stats.n_subsets,
            "split": stats.split,
            "f1": _round(stats.f1) if task_name in REPORT_MACRO_F1 else None,
            "accuracy_at_8": _round(k8_stats.mean) if k8_stats else None,
            "contamination": status,
            "contamination_reason": contamination_reason(model_id, task_name),
            "in_type_macro": _in_type_macro(
                task_name, task_type, status == "contaminated", lexical
            ),
            "in_by_type": _in_aggregates(
                task_name, task_type, status == "contaminated", lexical
            ),
        }
        scores.append(row)

    collapsed = _dataset_means_by_type(scores, flag="in_by_type")
    by_type: dict[str, Any] = {}
    type_macro_vals: list[float] = []
    for task_type in CATEGORY_ORDER:
        pairs = collapsed.get(task_type)
        if not pairs:
            continue
        chosen = _majority_metric([m for m, _ in pairs], task_type)
        vals = [v for m, v in pairs if m == chosen]
        if not vals:
            continue
        mu = mean(vals)
        sd = stdev(vals) if len(vals) >= 2 else None
        by_type[task_type] = {
            "metric": chosen,
            "value": _round(mu, 4),
            "std": _round(sd, 4),
            "n_datasets": len(vals),
        }
        if task_type in TYPE_MACRO_CATEGORIES:
            type_macro_vals.append(mu)

    task_macro_vals = [row["value"] for row in scores if row["in_type_macro"]]
    if not scores and not failed:
        status = "failed"
    elif failed and scores:
        status = "partial"
    elif failed:
        status = "failed"
    else:
        status = "succeeded"

    max_len = _max_seq_length(meta)
    summary: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": __version__,
        "status": status,
        "model": model_id,
        "job_id": job_id,
        "preset": preset,
        "max_seq_length": max_len,
        "lexical_baseline": lexical,
        "failed_tasks": failed,
        "n_tasks": len(scores),
        "scores": scores,
        "by_type": by_type,
        "type_macro": _round(mean(type_macro_vals), 4) if type_macro_vals else None,
        "task_macro": _round(mean(task_macro_vals), 4) if task_macro_vals else None,
        "overall_borda": None,
        "overall_note": OVERALL_NOTE,
    }
    return summary


def write_summary(
    results_dir: str | Path,
    summary_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    summary = build_summary(results_dir, **kwargs)
    dest = Path(summary_path) if summary_path else Path(results_dir) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    dest.write_text(payload, encoding="utf-8")
    print(f"[romteb] wrote {dest}  status={summary['status']}  n_tasks={summary['n_tasks']}")
    model = kwargs.get("model") or summary.get("model")
    if model:
        slug = str(model).replace("/", "__")
        metas = list((Path(results_dir) / slug).rglob("model_meta.json"))
        if metas:
            sidecar = max(metas, key=lambda p: p.stat().st_mtime).parent / "summary.json"
            if sidecar.resolve() != dest.resolve():
                sidecar.write_text(payload, encoding="utf-8")
                print(f"[romteb] wrote {sidecar}")
    return summary
