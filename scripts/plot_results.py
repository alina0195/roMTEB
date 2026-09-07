"""RoMTEB v1 figures from aggregator CSVs (per domain / task / model).

USAGE:
    ./apptainer-exec-romteb.sh scripts/plot_results.py --results_dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from romteb.eval_config import LEADERBOARD_EXCLUDE

# Distinct from the dense palette. BM25 is a lexical reference, not a
# ranked embedder — shown on Retrieval / Reranking figures only.
LEXICAL_COLOR = "#546E7A"

PALETTE = [
    "#2196F3",
    "#E91E63",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#00BCD4",
    "#F44336",
    "#8BC34A",
    "#795548",
    "#607D8B",
    "#3F51B5",
    "#009688",
    "#FF5722",
    "#673AB7",
    "#CDDC39",
    "#00ACC1",
    "#5C6BC0",
    "#EC407A",
    "#66BB6A",
    "#FFA726",
]

TYPE_ORDER = [
    "Classification",
    "PairClassification",
    "STS",
    "Retrieval",
    "Reranking",
]

SKIP_TASK_TYPES = {"Clustering", "Summarization"}
# Tasks that are also present as historical results but should never appear in
# plots. The four MCQ-as-retrieval variants were removed from the benchmark in
# Sep 2026 (`docs/task_audit.md`); listed here so that regenerating plots from
# an archived `results/` tree stays consistent with the current policy.
SKIP_TASKS = {
    "JuRoLegalExamRetrieval",
    "WWTBMRoQARetrieval",
    "RoMedQAv2Retrieval",
    "GrileGrammarRetrieval",
}

# Tasks kept in per-task tables (for transparency) but hidden from
# distribution / heatmap plots because they don't discriminate: saturated
# (BM25 solves them) or overall-excluded (all models below random). See
# `SATURATED_TASKS` and `OVERALL_EXCLUDE_TASKS` in `romteb/eval_config.py`.
AUDIT_EXCLUDED_TASKS = {
    "XQuADRetrieval",
    "JuRoLegalExamReranking",
}

DOMAIN_ORDER = [
    "Legal",
    "Medical",
    "Grammar",
    "Math",
    "Science",
    "Academic",
    "Reviews",
    "News",
    "Social",
    "Spoken",
    "Culture",
    "Encyclopaedic",
    "Web",
]

CMAP = LinearSegmentedColormap.from_list("romteb", ["#f7f7f7", "#2196F3", "#0d47a1"])


def short_name(model: str) -> str:
    name = str(model).split("/")[-1].split("__")[-1]
    return name if len(name) <= 28 else name[:25] + "…"


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] missing {path.name}")
        return None
    df = pd.read_csv(path)
    if "Model" in df.columns:
        df["Model"] = df["Model"].astype(str).str.replace("__", "/", regex=False)
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.replace("__", "/", regex=False)
    return df


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _numeric_wide(df: pd.DataFrame, id_col: str = "Model") -> pd.DataFrame:
    out = df.copy()
    out.replace(r"\s*[\*?]\s*$", "", regex=True, inplace=True)
    out.replace("-", np.nan, inplace=True)
    for col in out.columns:
        if col == id_col:
            continue
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.replace(r"\s*±.*$", "", regex=True)
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _color_map(models: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _lexical_models(cells: pd.DataFrame | None) -> list[str]:
    if cells is None or cells.empty or "lexical_baseline" not in cells.columns:
        return []
    return list(
        dict.fromkeys(cells.loc[_to_bool(cells["lexical_baseline"]), "model"].tolist())
    )


def _with_baselines(models: list[str], lexical: list[str]) -> list[str]:
    return list(models) + [m for m in lexical if m not in models]


def _grouped_bars(ax, models: list[str], categories: list[str], values: pd.DataFrame, cmap: dict[str, str]):
    n_models = max(len(models), 1)
    n_cats = len(categories)
    x = np.arange(n_cats)
    total = 0.82
    bar_w = total / n_models
    for i, model in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * bar_w
        vals = values.loc[model, categories].astype(float).values if model in values.index else np.full(n_cats, np.nan)
        ax.bar(
            x + offset,
            vals,
            width=bar_w * 0.9,
            color=cmap[model],
            label=short_name(model),
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=28, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7, ncol=2, framealpha=0.85)


def _heatmap(df: pd.DataFrame, title: str, out: Path, vmin: float = 0.0, vmax: float = 1.0) -> None:
    if df.empty:
        return
    n_r, n_c = df.shape
    fig_w = max(8.0, 0.55 * n_c + 3.5)
    fig_h = max(3.5, 0.42 * n_r + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(df.values, aspect="auto", cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(df.columns, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(df.index, fontsize=8)
    annotate = n_r * n_c <= 500
    if annotate:
        for r in range(n_r):
            for c in range(n_c):
                val = df.values[r, c]
                if np.isfinite(val):
                    ax.text(
                        c,
                        r,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if val > 0.62 else "black",
                    )
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    _save(fig, out)


def plot_overall_borda(summary: pd.DataFrame, out_dir: Path, cmap: dict[str, str]) -> None:
    rank_col = "Overall (Borda rank)"
    if rank_col not in summary.columns:
        print("[skip] no Borda column")
        return
    df = _numeric_wide(summary)
    df = df.dropna(subset=[rank_col]).sort_values(rank_col, ascending=False)
    if df.empty:
        return
    names = [short_name(m) for m in df["Model"]]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4 * len(df) + 1.5)))
    colors = [cmap.get(m, "#607D8B") for m in df["Model"]]
    bars = ax.barh(names, df[rank_col].values, color=colors, edgecolor="white", height=0.62)
    for bar, v in zip(bars, df[rank_col].values):
        ax.text(v + 0.03, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Borda rank (lower is better)")
    ax.set_title("RoMTEB – Overall (Borda rank)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    _save(fig, out_dir / "01_overall_borda.png")

    macro_col = "Overall (type-macro)"
    if macro_col in df.columns and df[macro_col].notna().any():
        df2 = df.dropna(subset=[macro_col]).sort_values(macro_col, ascending=True)
        names = [short_name(m) for m in df2["Model"]]
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4 * len(df2) + 1.5)))
        colors = [cmap.get(m, "#607D8B") for m in df2["Model"]]
        bars = ax.barh(names, df2[macro_col].values, color=colors, edgecolor="white", height=0.62)
        for bar, v in zip(bars, df2[macro_col].values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Type-macro (mean of Class / PairClass / Retrieval / Reranking)")
        ax.set_title("RoMTEB – Type-macro score (side column, not official Overall)", fontsize=13, fontweight="bold")
        ax.set_xlim(0, min(1.05, float(df2[macro_col].max()) * 1.25 + 0.05))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        _save(fig, out_dir / "02_type_macro.png")


def plot_by_task_type(summary: pd.DataFrame, out_dir: Path, models: list[str], cmap: dict[str, str]) -> None:
    df = _numeric_wide(summary).set_index("Model")
    cats = [c for c in TYPE_ORDER if c in df.columns and df[c].notna().any()]
    if not cats:
        return
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(cats) + 3), 6))
    _grouped_bars(ax, models, cats, df, cmap)
    ax.set_ylabel("Score")
    ax.set_title("RoMTEB – Scores by task type (dataset-macro; Reranking = acc@1)", fontsize=13, fontweight="bold")
    _save(fig, out_dir / "03_scores_by_task_type.png")


def plot_radar(summary: pd.DataFrame, out_dir: Path, models: list[str], cmap: dict[str, str]) -> None:
    df = _numeric_wide(summary).set_index("Model")
    cats = [c for c in TYPE_ORDER if c in df.columns and df[c].notna().any()]
    if len(cats) < 3:
        return
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    for model in models:
        if model not in df.index:
            continue
        vals = df.loc[model, cats].fillna(0).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=cmap[model], linewidth=2, label=short_name(model))
        ax.fill(angles, vals, color=cmap[model], alpha=0.06)
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("RoMTEB – Radar by task type", fontsize=13, fontweight="bold", pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.38, -0.05), fontsize=7)
    _save(fig, out_dir / "04_radar_task_types.png")


def plot_task_heatmap(cells: pd.DataFrame, out_dir: Path, models: list[str]) -> None:
    work = cells.copy()
    work = work[~work["task"].isin(SKIP_TASKS | AUDIT_EXCLUDED_TASKS)]
    work = work[~work["task_type"].isin(SKIP_TASK_TYPES | {"BitextMining"})]
    pivot = work.pivot_table(index="model", columns="task", values="mean", aggfunc="first")
    pivot = pivot.reindex([m for m in models if m in pivot.index])
    pivot.index = [short_name(m) for m in pivot.index]
    _heatmap(
        pivot,
        "RoMTEB – Per-task scores (BM25 on IR tasks; saturated/random-baseline tasks omitted)",
        out_dir / "05_per_task_heatmap.png",
    )


def plot_new_and_cross(results_dir: Path, out_dir: Path, models: list[str], cmap: dict[str, str]) -> None:
    for fname, title, out_name in (
        ("romteb_results_new_tasks.csv", "RoMTEB – New tasks", "06_new_tasks.png"),
        ("romteb_results_crosslingual.csv", "RoMTEB – Cross-lingual (BitextMining, not in Overall)", "07_crosslingual.png"),
        ("romteb_results_mcq_retrieval.csv", "RoMTEB – MCQ Retrieval diagnostic (not in Overall)", "08_mcq_retrieval.png"),
        ("romteb_results_by_dataset.csv", "RoMTEB – Per dataset × type", "09_by_dataset.png"),
    ):
        raw = _read_csv(results_dir / fname)
        if raw is None or raw.empty:
            continue
        df = _numeric_wide(raw).set_index("Model")
        cols = [c for c in df.columns if df[c].notna().any()]
        if not cols:
            continue
        if fname.endswith("by_dataset.csv"):
            _heatmap(
                df.loc[[m for m in models if m in df.index], cols].copy().rename(index=short_name),
                title,
                out_dir / out_name,
            )
            continue
        use_models = [m for m in models if m in df.index]
        fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(cols) + 3), 5.5))
        _grouped_bars(ax, use_models, cols, df, cmap)
        ax.set_ylabel("Score")
        ax.set_title(title, fontsize=12, fontweight="bold")
        _save(fig, out_dir / out_name)


def plot_domain_slices(
    slices: pd.DataFrame,
    out_dir: Path,
    models: list[str],
    cmap: dict[str, str],
    ir_models: list[str] | None = None,
) -> None:
    work = slices.copy()
    work["model"] = work["model"].astype(str)
    pivot = work.pivot_table(index="model", columns="slice", values="mean", aggfunc="first")
    ordered = [s for s in sorted(pivot.columns, key=lambda x: (DOMAIN_ORDER.index(x.split(" / ")[0]) if x.split(" / ")[0] in DOMAIN_ORDER else 99, x))]
    roster = ir_models or models
    pivot = pivot.reindex([m for m in roster if m in pivot.index])[ordered]
    _heatmap(
        pivot.rename(index=short_name),
        "RoMTEB – Domain × task type (one metric per column)",
        out_dir / "10_by_domain_heatmap.png",
    )

    for family, out_name, ylabel, title in (
        ("Retrieval", "11_retrieval_by_domain.png", "nDCG@10", "RoMTEB – Retrieval by domain"),
        ("Reranking", "12_reranking_by_domain.png", "primary metric", "RoMTEB – Reranking by domain"),
        ("Classification", "13_classification_by_domain.png", "accuracy", "RoMTEB – Classification by domain"),
    ):
        sub = work[work["task_type"] == family]
        if sub.empty:
            continue
        wide = sub.pivot_table(index="model", columns="domain", values="mean", aggfunc="first")
        cols = [d for d in DOMAIN_ORDER if d in wide.columns]
        cols += [c for c in wide.columns if c not in cols]
        family_models = (ir_models or models) if family in {"Retrieval", "Reranking"} else models
        use_models = [m for m in family_models if m in wide.index]
        if not use_models or not cols:
            continue
        fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(cols) + 3), 5.8))
        _grouped_bars(ax, use_models, cols, wide, cmap)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight="bold")
        _save(fig, out_dir / out_name)


def plot_cells_by_type(
    cells: pd.DataFrame,
    out_dir: Path,
    models: list[str],
    cmap: dict[str, str],
    ir_models: list[str] | None = None,
) -> None:
    work = cells.copy()
    ir_models = ir_models or models
    for ttype in TYPE_ORDER:
        sub = work[work["task_type"] == ttype]
        if ttype not in {"Retrieval", "Reranking"} and "lexical_baseline" in sub.columns:
            sub = sub[~_to_bool(sub["lexical_baseline"])]
        if sub.empty:
            continue
        wide = sub.pivot_table(index="model", columns="task", values="mean", aggfunc="first")
        cols = list(wide.columns)
        roster = ir_models if ttype in {"Retrieval", "Reranking"} else models
        use_models = [m for m in roster if m in wide.index]
        if not use_models:
            continue
        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(cols) + 3), 5.8))
        _grouped_bars(ax, use_models, cols, wide, cmap)
        ax.set_ylabel("Score")
        ax.set_title(f"RoMTEB – {ttype} per task", fontsize=13, fontweight="bold")
        slug = ttype.lower()
        _save(fig, out_dir / f"14_{slug}_per_task.png")


def plot_per_model_profiles(cells: pd.DataFrame, out_dir: Path, models: list[str], cmap: dict[str, str]) -> None:
    work = cells.copy()
    work = work[~work["task"].isin(SKIP_TASKS | AUDIT_EXCLUDED_TASKS)]
    work = work[~work["task_type"].isin(SKIP_TASK_TYPES | {"BitextMining"})]
    if work.empty:
        return
    n = len(models)
    if n == 0:
        return
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.2 * nrows), sharex=False)
    axes_list = np.atleast_1d(axes).ravel()
    tasks = sorted(work["task"].unique())
    for ax, model in zip(axes_list, models):
        sub = work[work["model"] == model].set_index("task").reindex(tasks)
        ax.barh([short_name(t) if False else t for t in tasks], sub["mean"].values, color=cmap[model], height=0.7)
        ax.set_title(short_name(model), fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis="y", labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    for ax in axes_list[n:]:
        ax.axis("off")
    fig.suptitle("RoMTEB – Per-model task profile", fontsize=13, fontweight="bold")
    _save(fig, out_dir / "15_per_model_task_profiles.png")


def plot_bootstrap(boot: pd.DataFrame, out_dir: Path, models: list[str], cmap: dict[str, str]) -> None:
    if boot.empty or not {"mean", "ci_lo", "ci_hi", "task", "model"}.issubset(boot.columns):
        return
    for task, sub in boot.groupby("task"):
        sub = sub.copy()
        sub["model"] = sub["model"].astype(str).str.replace("__", "/", regex=False)
        order = [m for m in models if m in set(sub["model"])] + [
            m for m in sub["model"] if m not in models
        ]
        seen = []
        for m in order:
            if m not in seen:
                seen.append(m)
        sub = sub.set_index("model").reindex(seen).dropna(subset=["mean"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(2.8, 0.38 * len(sub) + 1.2)))
        y = np.arange(len(sub))
        ax.hlines(y, sub["ci_lo"], sub["ci_hi"], color="#90A4AE", linewidth=2)
        for i, (model, row) in enumerate(sub.iterrows()):
            ax.plot(row["mean"], i, "o", color=cmap.get(model, "#455A64"), markersize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([short_name(m) for m in sub.index])
        ax.set_xlabel("per-query mean + 95% bootstrap CI")
        ax.set_title(f"IR bootstrap – {task}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        slug = "".join(ch if ch.isalnum() else "_" for ch in task)
        _save(fig, out_dir / f"16_bootstrap_{slug}.png")


def write_index(out_dir: Path) -> None:
    pngs = sorted(out_dir.glob("*.png"))
    lines = [
        "# RoMTEB plots",
        "",
        "Generated by `scripts/plot_results.py` from the aggregator CSVs.",
        "Official ranking is **Overall (Borda rank)** (lower is better).",
        "BM25 (`romteb/bm25s-ro`) is the lexical baseline on Retrieval / Reranking",
        "figures only — it does not enter Overall.",
        "Domain columns are one metric each — do not average Legal Retrieval with Legal Reranking.",
        "",
    ]
    for p in pngs:
        lines.append(f"- `{p.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'README.md'} ({len(pngs)} figures)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    summary = _read_csv(results_dir / "romteb_results_summary.csv")
    cells = _read_csv(results_dir / "romteb_results_cells.csv")
    slices = _read_csv(results_dir / "romteb_results_domain_slices.csv")
    boot = _read_csv(results_dir / "romteb_ir_bootstrap.csv")
    if boot is not None and not boot.empty and "model" in boot.columns:
        boot = boot[~boot["model"].isin(LEADERBOARD_EXCLUDE)].copy()

    models: list[str] = []
    if summary is not None and not summary.empty:
        rank_col = "Overall (Borda rank)"
        if rank_col in summary.columns:
            tmp = _numeric_wide(summary)
            tmp = tmp.dropna(subset=[rank_col]).sort_values(rank_col)
            models = tmp["Model"].tolist()
        else:
            models = summary["Model"].tolist()
    if cells is not None and "lexical_baseline" in cells.columns:
        models = [m for m in models if m not in set(cells.loc[_to_bool(cells["lexical_baseline"]), "model"])]
    models = [m for m in models if m not in LEADERBOARD_EXCLUDE]
    if not models and cells is not None:
        models = sorted(
            m for m in cells["model"].unique() if m not in LEADERBOARD_EXCLUDE
        )
    lexical = _lexical_models(cells)
    ir_models = _with_baselines(models, lexical)
    cmap = _color_map(models)
    for name in lexical:
        cmap[name] = LEXICAL_COLOR

    if summary is not None:
        plot_overall_borda(summary, out_dir, cmap)
        plot_by_task_type(summary, out_dir, models, cmap)
        plot_radar(summary, out_dir, models, cmap)
    if cells is not None:
        plot_task_heatmap(cells, out_dir, ir_models)
        plot_cells_by_type(cells, out_dir, models, cmap, ir_models=ir_models)
        plot_per_model_profiles(cells, out_dir, ir_models, cmap)
    plot_new_and_cross(results_dir, out_dir, ir_models, cmap)
    if slices is not None and not slices.empty:
        plot_domain_slices(slices, out_dir, models, cmap, ir_models=ir_models)
    if boot is not None and not boot.empty:
        plot_bootstrap(boot, out_dir, ir_models, cmap)

    write_index(out_dir)
    print(f"\nDone. All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
