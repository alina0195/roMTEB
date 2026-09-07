"""Aggregate MTEB result JSONs into statistically explicit tables.

USAGE:
    ./apptainer-exec-romteb.sh scripts/aggregate_results.py
    ./apptainer-exec-romteb.sh scripts/aggregate_results.py --results_dir results_new

Outputs (in <results_dir>/):
  - romteb_results_cells.csv                atomic (embedder, task) rows
  - romteb_results_by_task.csv / .md        model × task (mean ± std)
  - romteb_results_by_dataset.csv / .md     model × dataset × task-type
  - romteb_results_by_domain.csv / .md      model × (domain / task-type)
  - romteb_results_retrieval_by_domain.csv / .md
  - romteb_results_reranking_by_domain.csv / .md
  - romteb_results_summary.csv / .md        type means + Overall
  - romteb_results_new_tasks.csv / .md
  - romteb_results_crosslingual.csv / .md
  - romteb_results_mcq_retrieval.csv / .md  MCQ full-bank retrieval (not Overall)

Aggregation (see docs/evaluation_protocol.md §9):
  - Official Overall = category Borda then mean of category ranks.
  - Type-macro / task-macro are side columns. Type-macro averages majority-metric
    type means (never accuracy with MAP).
  - A category missing for any reported (non-lexical) model is dropped
    from Overall for everyone.
  - Collapse to dataset before domain/type means.
  - Confirmed contaminated cells are excluded from every mean.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

import pandas as pd

from romteb.bm25_ro import LEXICAL_BASELINE_NAMES
from romteb.contamination import (
    CONTAM_MARK,
    UNKNOWN_MARK,
    contamination_reason,
    contamination_status,
    iter_known_contaminated,
)
from romteb.eval_config import (
    CATEGORY_PRIMARY_METRIC,
    DOMAIN_ORDER,
    LEADERBOARD_EXCLUDE,
    MCQ_RETRIEVAL_TASKS,
    OVERALL_EXCLUDE_TASKS,
    REPORT_MACRO_F1,
    RERANKING_RANDOM_BASELINE,
    SATURATED_TASKS,
    dataset_of,
    metric_of,
    reporting_domain,
)
from romteb.scoring import (
    ScoreStats,
    category_borda_then_mean_rank,
    extract_score_stats,
    format_score,
    mean_rank_across_tasks,
    sample_mean_std,
    stderr,
)

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

CROSSLINGUAL_CATEGORIES = {"BitextMining"}

EXCLUDED_TASKS = {
    "FloresBitextMining",
    "XQuADRoRetrieval",
    "RoDTALLawsPairClassification",
    "MedQARoRetrieval",
    "Moroco.v2",
    "Moroco",
    "JuRoLegalExamPairClassification",
    "WWTBMRoQAPairClassification",
    "RoMedQAv2PairClassification",
    "MSMarcoRoRetrieval",
    "FakenewsRoClassification",
    "ROFFClassification",
    "RoOffenseSequencesClassification",
    "FBRoOffenseClassification",
}

TASK_RENAME = {
    "RonSTS": "RoSTS",
}

REUSED_TASK_NAMES = {
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "SIB200Classification",
    "RomanianReviewsSentiment.v2",
    "RomanianSentimentClassification.v2",
    "SIB200ClusteringS2S",
    "WebFAQRetrieval",
    "WikipediaRetrievalMultilingual",
    "XQuADRetrieval",
    "NTREXBitextMining",
    "Tatoeba",
    "IWSLT2017BitextMining",
}

try:
    from romteb.benchmark import ROMTEB_TASKS

    TASK_CATEGORY = {
        t.metadata.name: t.metadata.type
        for t in ROMTEB_TASKS
        if getattr(t, "metadata", None)
    }
except Exception:
    TASK_CATEGORY = {}

EXCLUDED_MODEL_SLUGS = {
    "readerbench__RoBERT-base",
    "readerbench__RoBERT-large",
    "HIT-TMG__KaLM-embedding-multilingual-mini-v1",
}


def _canon_model(slug: str) -> str:
    return slug.replace("__", "/")


def _is_lexical_baseline(model_slug: str) -> bool:
    name = _canon_model(model_slug)
    return name in LEXICAL_BASELINE_NAMES or model_slug in {
        n.replace("/", "__") for n in LEXICAL_BASELINE_NAMES
    }


def _is_reported_model(model_slug: str) -> bool:
    return not model_slug.startswith("_") and model_slug not in EXCLUDED_MODEL_SLUGS


def _is_overall_model(model_slug: str) -> bool:
    return _is_reported_model(model_slug) and not _is_lexical_baseline(model_slug)


def _infer_category(task_name: str, result: dict | None = None) -> str | None:
    if result:
        category = result.get("task_type")
        if category in CATEGORY_PRIMARY_METRIC:
            return category
    if task_name in TASK_CATEGORY:
        return TASK_CATEGORY[task_name]
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


def _walk_results(
    results_dir: Path,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """Map model_slug -> task_name -> result_dict. k8 sidecars are separate."""
    chosen: dict[str, dict[str, tuple[float, dict]]] = defaultdict(dict)
    k8: dict[str, dict[str, tuple[float, dict]]] = defaultdict(dict)
    for path in results_dir.rglob("*.json"):
        if path.name in {"model_meta.json", "_index.json"}:
            continue
        rel = path.relative_to(results_dir)
        top = rel.parts[0] if rel.parts else ""
        if top in {"predictions", "plots"} or top.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        is_k8 = path.name.endswith(".k8.json")
        stem = path.stem[: -len(".k8")] if is_k8 else path.stem
        task_name = data.get("task_name") or stem
        if is_k8 and isinstance(task_name, str) and task_name.endswith(".k8"):
            task_name = task_name[: -len(".k8")]
        task_name = TASK_RENAME.get(task_name, task_name)
        if task_name in EXCLUDED_TASKS:
            continue
        model_slug = _extract_model_slug(path, results_dir)
        mtime = path.stat().st_mtime
        bucket = k8 if is_k8 else chosen
        prev = bucket[model_slug].get(task_name)
        if prev is None or mtime >= prev[0]:
            bucket[model_slug][task_name] = (mtime, data)
    by_model = {
        model: {task: payload for task, (_mt, payload) in tasks.items()}
        for model, tasks in chosen.items()
    }
    k8_by_model = {
        model: {task: payload for task, (_mt, payload) in tasks.items()}
        for model, tasks in k8.items()
    }
    return by_model, k8_by_model


def _extract_model_slug(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if parts else "unknown"


def _model_display_name(slug: str) -> str:
    return slug.replace("__", "/")


def _read_model_meta(results_dir: Path, model_slug: str) -> dict:
    meta_paths = list((results_dir / model_slug).rglob("model_meta.json"))
    if not meta_paths:
        return {}
    newest = max(meta_paths, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(newest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _max_seq_length(meta: dict) -> int | None:
    for key in ("max_tokens", "max_seq_length", "romteb_max_seq_length"):
        v = meta.get(key)
        if isinstance(v, (int, float)) and v == v:
            return int(v)
    return None


def _load_ir_stats(results_dir: Path) -> dict[str, dict]:
    path = results_dir / "romteb_ir_task_stats.csv"
    if not path.exists():
        fallback = Path(__file__).resolve().parents[1] / "romteb_ir_task_stats.csv"
        path = fallback if fallback.exists() else path
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict] = {}
    for row in df.to_dict(orient="records"):
        out[str(row["task"])] = row
    return out


def _contamination_note() -> str:
    lines = [
        f"`{CONTAM_MARK.strip()}` confirmed contaminated: excluded from every mean "
        "(kept in the per-task table). "
        f"`{UNKNOWN_MARK.strip()}` training data unpublished: kept in means, flagged."
    ]
    for model, task_name, reason in iter_known_contaminated():
        lines.append(f"  - {model} / {task_name}: {reason}")
    return "\n".join(lines)


def _sorted_task_names(task_names: set[str]) -> list[str]:
    def sort_key(name: str) -> tuple:
        category = _infer_category(name) or "ZZZ"
        cat_idx = CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else 99
        return (cat_idx, name)

    return sorted(task_names, key=sort_key)


def _stats_for(result: dict, task_type: str | None, task_name: str) -> ScoreStats | None:
    key = metric_of(task_type or "", task_name)
    return extract_score_stats(result, metric_key=key)


def _k8_accuracy(k8_result: dict | None) -> float | None:
    if not k8_result:
        return None
    stats = extract_score_stats(k8_result, metric_key="accuracy")
    return None if stats is None else stats.mean


def _build_cells(
    by_model: dict[str, dict[str, dict]],
    k8_by_model: dict[str, dict[str, dict]],
    results_dir: Path,
    ir_stats: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for model_slug, tasks in sorted(by_model.items()):
        if not _is_reported_model(model_slug):
            continue
        meta = _read_model_meta(results_dir, model_slug)
        max_len = _max_seq_length(meta)
        for task_name, result in sorted(tasks.items()):
            task_type = _infer_category(task_name, result)
            stats = _stats_for(result, task_type, task_name)
            if stats is None:
                continue
            status = contamination_status(model_slug, task_name)
            contaminated = status == "contaminated"
            se = stderr(stats)
            ir = ir_stats.get(task_name, {})
            k8 = _k8_accuracy(k8_by_model.get(model_slug, {}).get(task_name))
            n_queries = ir.get("n_queries")
            n_corpus = ir.get("n_corpus")
            rows.append(
                {
                    "model": _model_display_name(model_slug),
                    "model_slug": model_slug,
                    "task": task_name,
                    "dataset": dataset_of(task_name),
                    "domain": reporting_domain(task_name) or "",
                    "task_type": task_type or "",
                    "metric": metric_of(task_type or "", task_name),
                    "split": stats.split,
                    "n": stats.n,
                    "n_subsets": stats.n_subsets,
                    "mean": round(stats.mean, 6),
                    "std": None if stats.std is None else round(stats.std, 6),
                    "stderr": None if se is None else round(se, 6),
                    "f1": None if stats.f1 is None else round(stats.f1, 6),
                    "accuracy": None if stats.accuracy is None else round(stats.accuracy, 6),
                    "map_at_1000": (
                        None if stats.map_at_1000 is None else round(stats.map_at_1000, 6)
                    ),
                    "accuracy_at_8": None if k8 is None else round(k8, 6),
                    "n_queries": n_queries if pd.notna(n_queries) else None,
                    "n_corpus": n_corpus if pd.notna(n_corpus) else None,
                    "max_seq_length": max_len,
                    "contaminated": contaminated,
                    "contamination_status": status,
                    "contamination_reason": contamination_reason(model_slug, task_name),
                    "lexical_baseline": _is_lexical_baseline(model_slug),
                    "in_overall": (
                        not contaminated
                        and (task_type or "") not in CROSSLINGUAL_CATEGORIES
                        and not _is_lexical_baseline(model_slug)
                    ),
                }
            )
    return rows


def _cells_df(cells: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(cells)


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df.astype(str).itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *body])


def _write_table(
    df: pd.DataFrame, csv_path: Path, md_path: Path, title: str, note: str = ""
) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False))
    df.to_csv(csv_path, index=False)
    md = _df_to_markdown(df) + "\n"
    if note:
        md += f"\n{note}\n"
    md_path.write_text(md)
    print(f"Wrote {csv_path}\nWrote {md_path}")


def _usable(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return cells
    return cells.loc[~cells["contaminated"].astype(bool)]


def _collapse_dataset_type(usable: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean of tasks that share (model, dataset, task_type, metric)."""
    if usable.empty:
        return usable
    rows = []
    grouped = usable.groupby(
        ["model", "model_slug", "dataset", "domain", "task_type", "metric"],
        dropna=False,
        sort=False,
    )
    for keys, grp in grouped:
        model, slug, dataset, domain, task_type, metric = keys
        vals = [float(v) for v in grp["mean"].tolist()]
        mu, sd, n = sample_mean_std(vals)
        rows.append(
            {
                "model": model,
                "model_slug": slug,
                "dataset": dataset,
                "domain": domain,
                "task_type": task_type,
                "metric": metric,
                "mean": mu,
                "std": sd,
                "n_tasks": n,
            }
        )
    return pd.DataFrame(rows)


def _wide_from_long(
    long_df: pd.DataFrame,
    *,
    col_key: str,
    col_order: list[str],
    value_col: str = "display",
) -> pd.DataFrame:
    models = sorted(long_df["model"].unique())
    cols = [c for c in col_order if c in set(long_df[col_key])]
    extra = sorted(set(long_df[col_key]) - set(cols))
    cols = cols + extra
    lookup = {
        (row["model"], row[col_key]): row[value_col]
        for row in long_df.to_dict(orient="records")
    }
    rows = []
    for model in models:
        row: dict[str, str] = {"Model": model}
        for col in cols:
            row[col] = lookup.get((model, col), "-")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["Model"] + [
        c for c in df.columns if c != "Model" and df[c].astype(str).ne("-").any()
    ]
    return df[keep]


def _cell_display(row: dict) -> str:
    display = format_score(
        ScoreStats(
            mean=float(row["mean"]),
            std=None if pd.isna(row["std"]) else float(row["std"]),
            n=int(row["n"]),
            split=str(row["split"]),
            f1=None if pd.isna(row.get("f1")) else float(row["f1"]),
        )
    )
    extras: list[str] = []
    if row["task"] in REPORT_MACRO_F1 and row.get("f1") is not None and pd.notna(row["f1"]):
        extras.append(f"F1={float(row['f1']):.4f}")
    if (
        row.get("accuracy_at_8") is not None
        and pd.notna(row.get("accuracy_at_8"))
        and row.get("task_type") == "Classification"
    ):
        extras.append(f"acc@8={float(row['accuracy_at_8']):.4f}")
    if extras:
        display = f"{display} ({', '.join(extras)})"
    status = row.get("contamination_status") or "clean"
    if status == "contaminated":
        display = f"{float(row['mean']):.4f}{CONTAM_MARK}"
    elif status == "unknown":
        display = f"{display}{UNKNOWN_MARK}"
    return display


def _build_task_matrix(cells: pd.DataFrame, task_names: list[str]) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    models = sorted(cells["model"].unique())
    lookup = {}
    for row in cells.to_dict(orient="records"):
        lookup[(row["model"], row["task"])] = _cell_display(row)
    rows = []
    for model in models:
        row: dict[str, str] = {"Model": model}
        for task in task_names:
            row[task] = lookup.get((model, task), "-")
        rows.append(row)
    return pd.DataFrame(rows)


def _domain_type_long(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Mean over datasets inside each (model, domain, task_type) sharing a metric."""
    if dataset_df.empty:
        return dataset_df
    keep = dataset_df[dataset_df["domain"].astype(str).str.len() > 0]
    keep = keep[keep["task_type"].isin(CATEGORY_PRIMARY_METRIC)]
    keep = keep[~keep["task_type"].isin(CROSSLINGUAL_CATEGORIES)]
    rows = []
    grouped = keep.groupby(
        ["model", "model_slug", "domain", "task_type", "metric"],
        dropna=False,
        sort=False,
    )
    for keys, grp in grouped:
        model, slug, domain, task_type, metric = keys
        vals = [float(v) for v in grp["mean"].tolist()]
        mu, sd, n = sample_mean_std(vals)
        slice_name = f"{domain} / {task_type}"
        rows.append(
            {
                "model": model,
                "model_slug": slug,
                "domain": domain,
                "task_type": task_type,
                "metric": metric,
                "slice": slice_name,
                "mean": mu,
                "std": sd,
                "n_datasets": n,
                "display": (
                    f"{mu:.4f}"
                    if sd is None
                    else f"{mu:.4f} ± {sd:.4f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _slice_order(slices: Iterable[str]) -> list[str]:
    def key(name: str) -> tuple:
        if " / " not in name:
            return (99, 99, name)
        domain, typ = name.split(" / ", 1)
        d_idx = DOMAIN_ORDER.index(domain) if domain in DOMAIN_ORDER else 99
        t_idx = CATEGORY_ORDER.index(typ) if typ in CATEGORY_ORDER else 99
        return (d_idx, t_idx, name)

    return sorted(set(slices), key=key)


def _majority_metric(metrics: list[str], task_type: str) -> str:
    """Pick one metric when a category mixes several.

    Majority vote across datasets. Ties prefer the category's declared
    primary, then Reranking accuracy@1 (the MCQ exam metric). Type-macro
    never averages accuracy with MAP — the minority metric stays in Borda
    and per-task tables only.
    """
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


def _type_means(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean over datasets inside each task type.

    Mixed-metric categories (MCQ accuracy vs RoMedQA MAP) keep the majority
    metric and drop the rest from the arithmetic mean. The omitted datasets
    still vote in Overall Borda.
    """
    if dataset_df.empty:
        return dataset_df
    keep = dataset_df[dataset_df["task_type"].isin(CATEGORY_PRIMARY_METRIC)]
    rows = []
    grouped = keep.groupby(
        ["model", "model_slug", "task_type"],
        dropna=False,
        sort=False,
    )
    for keys, grp in grouped:
        model, slug, task_type = keys
        metric_list = [str(m) for m in grp["metric"].tolist()]
        chosen = _majority_metric(metric_list, str(task_type))
        sub = grp[grp["metric"].astype(str) == chosen]
        vals = [float(v) for v in sub["mean"].tolist()]
        if not vals:
            rows.append(
                {
                    "model": model,
                    "model_slug": slug,
                    "task_type": task_type,
                    "metric": "mixed",
                    "mean": None,
                    "std": None,
                    "n_datasets": int(grp["dataset"].nunique()),
                }
            )
            continue
        mu, sd, n = sample_mean_std(vals)
        rows.append(
            {
                "model": model,
                "model_slug": slug,
                "task_type": task_type,
                "metric": chosen,
                "mean": mu,
                "std": sd,
                "n_datasets": n,
            }
        )
    return pd.DataFrame(rows)


def _overall_categories(type_df: pd.DataFrame, overall_models: list[str]) -> list[str]:
    """Categories valid for every reported non-lexical model."""
    if type_df.empty or not overall_models:
        return []
    cats = [c for c in CATEGORY_ORDER if c not in CROSSLINGUAL_CATEGORIES]
    present = set(zip(type_df["model"], type_df["task_type"]))
    valid: list[str] = []
    for cat in cats:
        if all((model, cat) in present for model in overall_models):
            valid.append(cat)
    return valid


def _summary_table(
    type_df: pd.DataFrame,
    usable_cells: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    models = sorted(type_df["model"].unique()) if not type_df.empty else []
    overall_models = [
        m
        for m in models
        if not _is_lexical_baseline(m.replace("/", "__"))
        and not _is_lexical_baseline(m)
    ]
    lookup = {
        (row["model"], row["task_type"]): row
        for row in type_df.to_dict(orient="records")
    }
    overall_cats = _overall_categories(
        type_df[type_df["model"].isin(overall_models)] if not type_df.empty else type_df,
        overall_models,
    )

    task_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row in usable_cells.to_dict(orient="records"):
        if row["task_type"] in CROSSLINGUAL_CATEGORIES:
            continue
        if row["task"] in MCQ_RETRIEVAL_TASKS:
            continue
        # Sep 2026 audit (`docs/task_audit.md`): saturated tasks (BM25 already
        # solves them) and tasks where every model is below the random
        # baseline don't discriminate. They stay in per-task tables so the
        # audit finding is visible, but never contribute to Overall / macros.
        if row["task"] in SATURATED_TASKS:
            continue
        if row["task"] in OVERALL_EXCLUDE_TASKS:
            continue
        if row.get("lexical_baseline"):
            continue
        task_scores[row["model"]][row["task"]] = float(row["mean"])
    all_tasks = sorted({t for tasks in task_scores.values() for t in tasks})
    ranks = mean_rank_across_tasks(task_scores, all_tasks)

    dataset_scores: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    if not dataset_df.empty:
        for row in dataset_df.to_dict(orient="records"):
            if row["task_type"] in CROSSLINGUAL_CATEGORIES:
                continue
            if _is_lexical_baseline(str(row["model_slug"])) or _is_lexical_baseline(
                str(row["model"])
            ):
                continue
            dataset_scores[row["model"]][(row["task_type"], row["dataset"])] = float(
                row["mean"]
            )
    borda = category_borda_then_mean_rank(dataset_scores, overall_cats)

    max_len_by_model: dict[str, str] = {}
    if "max_seq_length" in usable_cells.columns:
        for row in usable_cells.to_dict(orient="records"):
            v = row.get("max_seq_length")
            if v is not None and pd.notna(v):
                max_len_by_model[row["model"]] = str(int(v))

    rows = []
    for model in models:
        row: dict[str, str | float] = {"Model": model}
        type_vals: list[float] = []
        n_types_shown = 0
        for cat in CATEGORY_ORDER:
            rec = lookup.get((model, cat))
            if rec is None:
                row[cat] = "-"
                continue
            n_types_shown += 1
            if rec["mean"] is None or (
                isinstance(rec["mean"], float) and pd.isna(rec["mean"])
            ):
                row[cat] = "mixed"
                continue
            row[cat] = f"{float(rec['mean']):.4f}"
        for cat in overall_cats:
            rec = lookup.get((model, cat))
            if rec is None or rec["mean"] is None:
                continue
            if isinstance(rec["mean"], float) and pd.isna(rec["mean"]):
                continue
            if rec.get("metric") == "mixed":
                continue
            type_vals.append(float(rec["mean"]))
        lexical = _is_lexical_baseline(model) or _is_lexical_baseline(
            model.replace("/", "__")
        )
        task_vals = list(task_scores.get(model, {}).values())
        row["n_types"] = n_types_shown
        row["n_overall_types"] = 0 if lexical else len(overall_cats)
        row["n_tasks"] = len(task_vals)
        row["max_seq_length"] = max_len_by_model.get(model, "-")
        if lexical:
            row["Overall (Borda rank)"] = "-"
            row["Overall (type-macro)"] = "-"
            row["Overall (task-macro)"] = "-"
            row["Mean rank"] = "-"
        else:
            row["Overall (Borda rank)"] = (
                f"{borda[model][0]:.2f}" if model in borda else "-"
            )
            row["Overall (type-macro)"] = (
                f"{mean(type_vals):.4f}" if type_vals else "-"
            )
            row["Overall (task-macro)"] = (
                f"{mean(task_vals):.4f}" if task_vals else "-"
            )
            if model in ranks:
                mu, _n = ranks[model]
                row["Mean rank"] = f"{mu:.2f}"
            else:
                row["Mean rank"] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def _dataset_wide(dataset_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_df.empty:
        return dataset_df
    work = dataset_df.copy()
    work["slice"] = work.apply(
        lambda r: f"{r['dataset']} / {r['task_type']}", axis=1
    )
    work["display"] = work.apply(
        lambda r: (
            f"{float(r['mean']):.4f}"
            if r["std"] is None or (isinstance(r["std"], float) and pd.isna(r["std"]))
            else f"{float(r['mean']):.4f} ± {float(r['std']):.4f}"
        ),
        axis=1,
    )

    def col_key(name: str) -> tuple:
        dataset, typ = name.split(" / ", 1)
        domain = ""
        hit = dataset_df[dataset_df["dataset"] == dataset]
        if not hit.empty:
            domain = str(hit.iloc[0]["domain"])
        d_idx = DOMAIN_ORDER.index(domain) if domain in DOMAIN_ORDER else 99
        t_idx = CATEGORY_ORDER.index(typ) if typ in CATEGORY_ORDER else 99
        return (d_idx, dataset, t_idx)

    order = sorted(work["slice"].unique(), key=col_key)
    return _wide_from_long(work, col_key="slice", col_order=order)


def _ir_size_note(cells: pd.DataFrame, family: str) -> str:
    sub = cells[cells["task_type"] == family]
    if sub.empty:
        return ""
    lines = [f"{family} task sizes (`n_queries` / `n_corpus`):"]
    seen = set()
    for row in sub.to_dict(orient="records"):
        task = row["task"]
        if task in seen:
            continue
        seen.add(task)
        nq, nc = row.get("n_queries"), row.get("n_corpus")
        nq_s = "-" if nq is None or (isinstance(nq, float) and pd.isna(nq)) else str(int(nq))
        nc_s = "-" if nc is None or (isinstance(nc, float) and pd.isna(nc)) else str(int(nc))
        lines.append(f"  - {task}: {nq_s} queries / {nc_s} docs")
    if family == "Reranking":
        lines.append("Random baselines (closed-set, one relevant):")
        for task, spec in RERANKING_RANDOM_BASELINE.items():
            if spec["k"] == 0:
                lines.append(
                    f"  - {task}: multi-relevant; MAP is the primary metric, no single acc@1 chance."
                )
            else:
                lines.append(
                    f"  - {task}: {int(spec['k'])} options; "
                    f"E[acc@1]={spec['accuracy']:.3f}, E[MAP]={spec['map_at_1000']:.3f}"
                )
    return "\n".join(lines)


def aggregate(results_dir: str, models: list[str] | None = None):
    root = Path(results_dir)
    if not root.exists():
        raise SystemExit(f"results dir not found: {root}")

    by_model, k8_by_model = _walk_results(root)
    if models:
        wanted = {m.replace("/", "__") for m in models}
        by_model = {s: t for s, t in by_model.items() if s in wanted or s.replace("__", "/") in models}
        k8_by_model = {s: t for s, t in k8_by_model.items() if s in by_model}
        if not by_model:
            raise SystemExit(f"no JSON results for --models {models} under {root}")
    else:
        drop = {m.replace("/", "__") for m in LEADERBOARD_EXCLUDE}
        skipped = sorted(s for s in by_model if s in drop)
        if skipped:
            print(f"[aggregate] excluding incomplete models: {skipped}")
        by_model = {s: t for s, t in by_model.items() if s not in drop}
        k8_by_model = {s: t for s, t in k8_by_model.items() if s in by_model}
    if not by_model:
        raise SystemExit(f"no JSON results under {root}")

    ir_stats = _load_ir_stats(root)
    cells = _build_cells(by_model, k8_by_model, root, ir_stats)
    if not cells:
        raise SystemExit(f"no usable scores under {root}")
    cells_df = _cells_df(cells)
    cells_csv = root / "romteb_results_cells.csv"
    cells_df.to_csv(cells_csv, index=False)
    print(f"Wrote {cells_csv} ({len(cells_df)} cells)")

    usable = _usable(cells_df)
    _leaderboard_exclude = MCQ_RETRIEVAL_TASKS | SATURATED_TASKS | OVERALL_EXCLUDE_TASKS
    leaderboard = (
        usable.loc[~usable["task"].isin(_leaderboard_exclude)]
        if not usable.empty
        else usable
    )
    dataset_df = _collapse_dataset_type(leaderboard)
    type_df = _type_means(dataset_df)
    domain_long = _domain_type_long(dataset_df)

    overall_models = sorted(
        {
            row["model"]
            for row in leaderboard.to_dict(orient="records")
            if not row.get("lexical_baseline")
        }
    )
    overall_cats = _overall_categories(
        type_df[type_df["model"].isin(overall_models)] if not type_df.empty else type_df,
        overall_models,
    )
    summary_note = (
        "Official Overall is **Borda rank** (lower is better): Borda inside each "
        "category (each dataset is a voter after source collapse), then the "
        "unweighted mean of those category ranks. This is a documented deviation "
        "from MMTEB, which Bordas over tasks and would overweight Classification. "
        f"Overall uses the intersection of categories valid for every reported "
        f"dense model: {', '.join(overall_cats) or '(none)'}. "
        "`n_overall_types` is that intersection size; `n_types` is how many "
        "category columns the model actually has. Type-macro / task-macro are "
        "side columns. Mixed-metric categories keep the majority metric "
        "(Reranking = accuracy@1 over WWTBM + GRILE; RoMedQA MAP votes in "
        "Borda only) so type-macro is Classification + PairClassification + "
        "Retrieval + Reranking. BitextMining and BM25 "
        "never contribute to Overall. Tasks flagged saturated (BM25 already "
        "solves them, e.g. XQuADRetrieval) or overall-excluded (every model "
        "≤ empirical random baseline, e.g. JuRoLegalExamReranking) also never "
        "contribute to Overall — see `docs/task_audit.md`. "
        "Confirmed contaminated cells are dropped from every mean. "
        "Classification ± is sample std over 10 LR probes; clustering "
        "± is MTEB's bootstrap `v_measure_std`. Metrics are never mixed in a mean. "
        "`max_seq_length` is the native window used (no common cap)."
    )
    summary = _summary_table(type_df, leaderboard, dataset_df)
    _write_table(
        summary,
        root / "romteb_results_summary.csv",
        root / "romteb_results_summary.md",
        "Category means (dataset-macro) + Overall",
        note=summary_note,
    )

    if not domain_long.empty:
        domain_long.to_csv(root / "romteb_results_domain_slices.csv", index=False)
        print(f"Wrote {root / 'romteb_results_domain_slices.csv'}")
        domain_wide = _wide_from_long(
            domain_long,
            col_key="slice",
            col_order=_slice_order(domain_long["slice"]),
        )
        domain_note = (
            "Each column is one application domain restricted to one task type "
            "(one metric). Cell = unweighted mean over **datasets** in that "
            "slice. `±` is the sample std across datasets (undefined when n=1). "
            "Legal / Retrieval = RoD-TAL nDCG@10. Legal / Reranking = JuRo "
            "accuracy@1 (MAP kept as a side column in cells). "
            "Medical / Reranking = RoMedQA MAP@1000. Mapping: `romteb/eval_config.py`."
        )
        _write_table(
            domain_wide,
            root / "romteb_results_by_domain.csv",
            root / "romteb_results_by_domain.md",
            "Application domain × task type (dataset-macro, one metric per column)",
            note=domain_note,
        )
        for family, fname, title in (
            (
                "Retrieval",
                "romteb_results_retrieval_by_domain",
                "Retrieval nDCG@10 by application domain",
            ),
            (
                "Reranking",
                "romteb_results_reranking_by_domain",
                "Reranking by application domain (primary metric per task)",
            ),
        ):
            sub = domain_long[domain_long["task_type"] == family]
            if sub.empty:
                continue
            wide = _wide_from_long(
                sub.assign(slice=sub["domain"]),
                col_key="slice",
                col_order=list(DOMAIN_ORDER),
            )
            extra = _ir_size_note(leaderboard, family)
            _write_table(
                wide,
                root / f"{fname}.csv",
                root / f"{fname}.md",
                title,
                note=domain_note + f" Only {family}.\n" + extra,
            )

    if not dataset_df.empty:
        dataset_df.to_csv(root / "romteb_results_dataset_slices.csv", index=False)
        print(f"Wrote {root / 'romteb_results_dataset_slices.csv'}")
    dataset_wide = _dataset_wide(dataset_df)
    if not dataset_wide.empty:
        _write_table(
            dataset_wide,
            root / "romteb_results_by_dataset.csv",
            root / "romteb_results_by_dataset.md",
            "Per-dataset scores (one column per dataset × task type)",
            note=(
                "Columns are source datasets, split by task type when a dataset "
                "hosts more than one protocol. MASSIVE intent and scenario are "
                "averaged into MASSIVE / Classification. RORetrieval outlet+type "
                "clustering collapse to one Clustering dataset. Same metric "
                "within a column."
            ),
        )

    all_task_names = _sorted_task_names(set(cells_df["task"]))
    contam_note = _contamination_note()
    task_note = (
        "Atomic scores: one cell per (embedder, task). Classification = mean "
        "± sample std over 10 few-shot LR probes; `acc@8` is the MTEB-default "
        "budget when the tuned k differs. RoABSA also prints macro-F1. "
        "Bitext = mean ± sample std over language-pair subsets. Clustering ± "
        "is bootstrap `v_measure_std`. Other types are a single split score. "
        + contam_note
    )
    if all_task_names:
        task_df = _build_task_matrix(cells_df, all_task_names)
        _write_table(
            task_df,
            root / "romteb_results_by_task.csv",
            root / "romteb_results_by_task.md",
            "Per-task scores (all tasks)",
            note=task_note,
        )

    new_task_names = [
        t
        for t in all_task_names
        if t not in REUSED_TASK_NAMES and t not in EXCLUDED_TASKS
    ]
    if new_task_names:
        new_df = _build_task_matrix(cells_df, new_task_names)
        _write_table(
            new_df,
            root / "romteb_results_new_tasks.csv",
            root / "romteb_results_new_tasks.md",
            "Per-task scores (new datasets)",
            note=task_note,
        )

    crosslingual_tasks = [
        t
        for t in all_task_names
        if _infer_category(t) in CROSSLINGUAL_CATEGORIES
    ]
    if crosslingual_tasks:
        cross_df = _build_task_matrix(cells_df, crosslingual_tasks)
        _write_table(
            cross_df,
            root / "romteb_results_crosslingual.csv",
            root / "romteb_results_crosslingual.md",
            "Cross-lingual section (BitextMining; excluded from Overall)",
            note=(
                "Mean ± sample std over all Romanian language-pair subsets in "
                "the result JSON. Reported separately; NOT included in Overall."
            ),
        )

    mcq_ret = [t for t in all_task_names if t in MCQ_RETRIEVAL_TASKS]
    if mcq_ret:
        mcq_df = _build_task_matrix(cells_df, mcq_ret)
        extra = _ir_size_note(
            cells_df[cells_df["task"].isin(MCQ_RETRIEVAL_TASKS)]
            if not cells_df.empty
            else cells_df,
            "Retrieval",
        )
        _write_table(
            mcq_df,
            root / "romteb_results_mcq_retrieval.csv",
            root / "romteb_results_mcq_retrieval.md",
            "MCQ-as-Retrieval (full option bank; excluded from Overall)",
            note=(
                "Same exams as Reranking, searched over the whole option bank. "
                "nDCG@10 is noisy when the gold option text is duplicated as a "
                "distractor for another question (`scripts/mcq_duplicate_stats.py`). "
                "Official exam protocol remains Reranking. " + extra
            ),
        )

    # Saturated / overall-excluded tasks: kept in per-task heatmaps for
    # transparency but broken out into their own descriptive table so the
    # audit reasoning is auditable. See `docs/task_audit.md`.
    audit_excluded = [
        t
        for t in all_task_names
        if t in SATURATED_TASKS or t in OVERALL_EXCLUDE_TASKS
    ]
    if audit_excluded:
        audit_df = _build_task_matrix(cells_df, audit_excluded)
        _write_table(
            audit_df,
            root / "romteb_results_audit_excluded.csv",
            root / "romteb_results_audit_excluded.md",
            "Audit-excluded tasks (saturated or below random baseline)",
            note=(
                "These tasks are shown here for transparency but never contribute "
                "to Overall / type-macro / domain aggregates. See "
                "`docs/task_audit.md` for the discrimination classification: "
                "saturated tasks are already solved by BM25; overall-excluded "
                "tasks sit at or below the empirical random baseline for every "
                "dense model."
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only include these HF ids (or slugs). Default: every model under results_dir.",
    )
    args = parser.parse_args()
    aggregate(args.results_dir, models=args.models)


if __name__ == "__main__":
    main()
