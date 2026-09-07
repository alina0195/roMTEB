"""Task discrimination audit for RoMTEB.

Diagnoses which tasks in the current roster actually separate models. For each
task computes:

  - n_models: how many rostered models have a score
  - mean, min, max, spread (max - min)
  - median, q25, q75, IQR
  - BM25 baseline (romteb/bm25s-ro) if available
  - Random baseline (from RERANKING_RANDOM_BASELINE) if available
  - Best model above BM25 by ≥0.05 vs BM25 (severe gap)
  - Best model above random baseline

Then classifies each task into one of:

  - `saturated`: BM25 already >= 0.85 AND max_dense - BM25 < 0.05
    → task is trivially lexical, drop or mark _saturated
  - `dead`: all models within ±0.02 of random baseline (or all models
    below random with spread < 0.05) → task doesn't discriminate
  - `borderline`: spread between 25th and 75th percentile < 0.05
    → weak discrimination
  - `discriminative`: spread ≥ 0.05 AND best clearly above BM25/random

Usage:
  ./apptainer-exec-romteb.sh python3 scripts/audit_tasks.py \\
     [--results results] [--md docs/task_audit.md] [--csv results/task_audit.csv]

Reads main_score per (model, task) — same convention as compare_bm25.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from romteb.eval_config import RERANKING_RANDOM_BASELINE  # noqa: E402
from romteb.bm25_ro import LEXICAL_BASELINE_NAMES  # noqa: E402

BM25_SLUG = "romteb__bm25s-ro"
BM25_SLUGS = {n.replace("/", "__") for n in LEXICAL_BASELINE_NAMES}

# Thresholds for the automatic classification. Values below are heuristics —
# a human reviewer should still look at each task before dropping it.
SATURATED_BM25_FLOOR = 0.85
SATURATED_GAP_CEIL = 0.05
DEAD_SPREAD_CEIL = 0.05
DEAD_RANDOM_BAND = 0.02
BORDERLINE_IQR_CEIL = 0.05


def _parse_result(json_path: Path) -> tuple[float | None, str]:
    """Return (main_score, task_type). Reads the JSON once."""
    try:
        with json_path.open() as fp:
            d = json.load(fp)
    except (json.JSONDecodeError, OSError):
        return None, ""
    task_type = str(d.get("task_type", ""))
    scores = d.get("scores", {})
    if not isinstance(scores, dict):
        return None, task_type
    for _, entries in scores.items():
        if not entries:
            continue
        e = entries[0] if isinstance(entries, list) else entries
        ms = e.get("main_score")
        if ms is not None:
            return float(ms), task_type
    return None, task_type


_TASK_TYPE_SUFFIXES = (
    "Reranking",
    "Retrieval",
    "PairClassification",
    "Classification",
    "Clustering",
    "STS",
    "BitextMining",
    "Summarization",
)


def _task_type_fallback(task_name: str) -> str:
    for suf in _TASK_TYPE_SUFFIXES:
        if task_name.endswith(suf):
            return suf
    return "Unknown"


SKIP_DIR_NAMES = {"predictions"}  # 100+MB ranked-list dumps, not score files


def _collect(
    results_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Return ({task_name: {model_slug: score}}, {task_name: task_type}).

    Skips top-level `predictions/` because it holds per-query ranked-list JSON
    dumps that can be hundreds of MB each and don't carry `main_score`.
    """
    out: dict[str, dict[str, float]] = defaultdict(dict)
    task_type: dict[str, str] = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in SKIP_DIR_NAMES:
            continue
        for ver in model_dir.iterdir():
            if not ver.is_dir():
                continue
            for f in ver.glob("*.json"):
                name = f.stem
                if name == "model_meta" or name.endswith("_predictions"):
                    continue
                s, tt = _parse_result(f)
                if s is None:
                    continue
                out[name][model_dir.name] = s
                if tt and name not in task_type:
                    task_type[name] = tt
    for name in out:
        if name not in task_type:
            task_type[name] = _task_type_fallback(name)
    return out, task_type


def _percentile(sorted_scores: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0,100]). Falls back to min/max at edges."""
    if not sorted_scores:
        return float("nan")
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    idx = (p / 100.0) * (len(sorted_scores) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_scores) - 1)
    frac = idx - lo
    return sorted_scores[lo] + frac * (sorted_scores[hi] - sorted_scores[lo])


def _classify(
    task_name: str,
    task_type: str,
    dense_scores: list[float],
    bm25: float | None,
    random_baseline: float | None,
) -> tuple[str, str]:
    """Return (label, reason)."""
    if not dense_scores:
        return "no-data", "no dense model scores"
    lo, hi = min(dense_scores), max(dense_scores)
    spread = hi - lo
    mean = sum(dense_scores) / len(dense_scores)

    # p10-p90 spread ignores the 1-2 mis-prompted outliers (embeddinggemma /
    # nemotron / bge-multilingual-gemma2 systematically pull the min down on
    # retrieval before the A2 fix). This is what actually matters for
    # discrimination among correctly-loaded models.
    sorted_scores = sorted(dense_scores)
    p10 = _percentile(sorted_scores, 10.0)
    p90 = _percentile(sorted_scores, 90.0)
    inner_spread = p90 - p10

    # saturated: BM25 already high AND dense doesn't add much (compare p90 to
    # BM25 too, since p90 is a robust "top of the pack" estimate).
    if bm25 is not None and bm25 >= SATURATED_BM25_FLOOR:
        gap_over_bm25 = hi - bm25
        p90_gap = p90 - bm25
        if gap_over_bm25 < SATURATED_GAP_CEIL and p90_gap < SATURATED_GAP_CEIL:
            return (
                "saturated",
                f"BM25={bm25:.3f} >= {SATURATED_BM25_FLOOR}; "
                f"best dense - BM25 = {gap_over_bm25:+.3f}, "
                f"p90 - BM25 = {p90_gap:+.3f} (< {SATURATED_GAP_CEIL})",
            )

    # dead: all around random baseline or all under random with tiny spread
    if random_baseline is not None:
        under_band = all(
            abs(s - random_baseline) < DEAD_RANDOM_BAND for s in dense_scores
        )
        if under_band and spread < DEAD_SPREAD_CEIL:
            return (
                "dead",
                f"all {len(dense_scores)} models within ±{DEAD_RANDOM_BAND} of "
                f"random={random_baseline:.3f}; spread={spread:.3f}",
            )
        all_under = all(s < random_baseline for s in dense_scores)
        if all_under and inner_spread < DEAD_SPREAD_CEIL:
            return (
                "dead",
                f"all {len(dense_scores)} models below random={random_baseline:.3f}; "
                f"inner spread p90-p10={inner_spread:.3f}",
            )

    # near-zero MCQ-retrieval collapse: p90 (top pack) still below 0.10
    if task_type == "Retrieval" and p90 < 0.10 and inner_spread < 0.05:
        return (
            "dead",
            f"top pack below 0.10 (p90={p90:.3f}, inner spread={inner_spread:.3f}); "
            "corpus duplicated surface forms — use *Reranking variant",
        )

    # borderline: inner-spread p10-p90 is what matters after excluding
    # broken-prompt outliers. If the middle 80% of models are bunched
    # within 0.05, the ranking noise dominates.
    if inner_spread < BORDERLINE_IQR_CEIL:
        return (
            "borderline",
            f"p90-p10={inner_spread:.3f} < {BORDERLINE_IQR_CEIL}; "
            f"middle 80% bunched near mean={mean:.3f} (full spread={spread:.3f})",
        )

    return (
        "discriminative",
        f"p90-p10={inner_spread:.3f}, full spread={spread:.3f}, mean={mean:.3f}",
    )


def _render_md(rows: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# RoMTEB — Task discrimination audit\n")
    lines.append(
        "Automatic classification of every task in the current roster based on "
        "how well models spread on it. Run with `python3 scripts/audit_tasks.py`. "
        "Thresholds live in `scripts/audit_tasks.py`.\n"
    )
    lines.append("## Policy applied to the current benchmark\n")
    lines.append(
        "This audit is the single source of truth for which tasks contribute "
        "to Overall / type-macro / domain slices. Concrete actions taken after "
        "the Sep 2026 review (also cross-referenced in `docs/evaluation_protocol.md`):\n"
    )
    lines.append(
        "- **Removed completely** (task classes deleted, HF datasets deleted, "
        "revision pins removed, historical results moved to "
        "`archive/2026-09-01-mcq-retrieval-removed/`): the four full-option-bank "
        "MCQ retrieval tasks — `GrileGrammarRetrieval`, `JuRoLegalExamRetrieval`, "
        "`RoMedQAv2Retrieval`, `WWTBMRoQARetrieval`. They collapsed to p90 "
        "nDCG@10 ≤ 0.18 because the option-bank corpus has duplicate surface "
        "forms (22–30% dup rate — see `scripts/mcq_duplicate_stats.py`). Use "
        "the `*Reranking` counterparts.\n"
        "- **Marked `[saturated]`, excluded from Overall** (kept in per-task "
        "tables for transparency, `SATURATED_TASKS` in `romteb/eval_config.py`): "
        "`XQuADRetrieval` — BM25 = 0.961 nDCG@10, best dense (e5-large) 0.975. "
        "Task is trivially lexical.\n"
        "- **Excluded from Overall** (kept in per-task tables, "
        "`OVERALL_EXCLUDE_TASKS`): `JuRoLegalExamReranking` — every dense "
        "model is below the empirical random baseline (0.319–0.358 acc@1 vs "
        "0.381 random; 0.363–0.410 MAP vs 0.639 random). Task measures legal "
        "domain knowledge; cosine similarity can't solve it.\n"
        "- **Kept, monitored**: 6 borderline tasks. `RoNLIPairClassification` "
        "will be re-evaluated after re-runs on the rebalanced test split "
        "(A1 fix); the rest have narrow inner spreads but are close to full "
        "saturation (RoABSA, SciTechBanRO) or already at the top of the "
        "possible range (RoMedQAv2Reranking).\n"
    )
    lines.append("## Classification labels\n")
    lines.append(
        "**Labels:**\n"
        "- `saturated` — BM25 already ≥0.85 and both `max` and `p90` are "
        "within 0.05 of BM25. The task is trivially lexical; dense adds no "
        "signal.\n"
        "- `dead` — every model is within ±0.02 of the empirical random "
        "baseline (or all under it with tiny p10-p90 spread), or the top "
        "pack (p90) is still below 0.10 on a Retrieval task (MCQ-as-retrieval "
        "collapse).\n"
        "- `borderline` — the inner spread (p90 − p10) is <0.05. Ignoring the "
        "1-2 mis-prompted outliers, the middle 80% of models are bunched too "
        "tightly for ranking noise to be trusted.\n"
        "- `discriminative` — real spread across models; keep.\n"
    )
    lines.append(
        "**Why p10-p90 instead of min-max?** Before the A2 prompt fixes, "
        "embeddinggemma, nemotron, and bge-multilingual-gemma2 were "
        "systematic outliers on retrieval (below BM25). Their scores blow "
        "up min-max but say nothing about task quality — the task can still "
        "be dead even if 2 broken models drag the spread to 0.5.\n"
    )
    lines.append(
        "| Task | Type | n | BM25 | Random | min | p10 | median | p90 | max | inner spread | Label | Reason |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")

    def _fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.3f}"

    for r in rows:
        lines.append(
            f"| `{r['task']}` | {r['type']} | {r['n']} | "
            f"{_fmt(r['bm25'])} | {_fmt(r['random'])} | "
            f"{_fmt(r['min'])} | {_fmt(r['p10'])} | {_fmt(r['median'])} | "
            f"{_fmt(r['p90'])} | {_fmt(r['max'])} | "
            f"{_fmt(r['inner_spread'])} | **{r['label']}** | {r['reason']} |"
        )
    lines.append("")

    # Grouped view
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    lines.append("\n## Grouped by verdict\n")
    for label in ("saturated", "dead", "borderline", "discriminative", "no-data"):
        items = by_label.get(label, [])
        if not items:
            continue
        lines.append(f"### {label} ({len(items)})\n")
        for r in sorted(items, key=lambda x: x["task"]):
            lines.append(f"- `{r['task']}` — {r['reason']}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_csv(rows: list[dict], out_path: Path) -> None:
    hdr = [
        "task", "type", "n", "bm25", "random",
        "min", "p10", "median", "p90", "max",
        "spread", "inner_spread",
        "label", "reason",
    ]
    lines = ["\t".join(hdr)]
    for r in rows:
        lines.append(
            "\t".join(
                str(r.get(k, ""))
                if not isinstance(r.get(k), float)
                else f"{r[k]:.4f}"
                for k in hdr
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--md", type=Path, default=Path("docs/task_audit.md"))
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    task_scores, task_types = _collect(args.results)
    if not task_scores:
        sys.exit(f"No results found under {args.results}")

    rows: list[dict] = []
    for task_name, model_scores in sorted(task_scores.items()):
        bm25 = None
        for slug in BM25_SLUGS:
            if slug in model_scores:
                bm25 = model_scores[slug]
                break
        dense_only = {m: s for m, s in model_scores.items() if m not in BM25_SLUGS}
        if not dense_only:
            continue

        task_type = task_types.get(task_name, _task_type_fallback(task_name))
        random_baseline = None
        if task_name in RERANKING_RANDOM_BASELINE:
            rb = RERANKING_RANDOM_BASELINE[task_name]
            random_baseline = rb.get("map_at_1000") or rb.get("accuracy")

        scores = list(dense_only.values())
        median = statistics.median(scores) if scores else None
        sorted_scores = sorted(scores)
        p10 = _percentile(sorted_scores, 10.0)
        p90 = _percentile(sorted_scores, 90.0)
        label, reason = _classify(task_name, task_type, scores, bm25, random_baseline)
        rows.append(
            {
                "task": task_name,
                "type": task_type,
                "n": len(scores),
                "bm25": bm25,
                "random": random_baseline,
                "min": min(scores),
                "p10": p10,
                "median": median,
                "p90": p90,
                "max": max(scores),
                "spread": max(scores) - min(scores),
                "inner_spread": p90 - p10,
                "label": label,
                "reason": reason,
            }
        )

    rows.sort(key=lambda r: (r["type"], r["task"]))

    args.md.parent.mkdir(parents=True, exist_ok=True)
    _render_md(rows, args.md)
    print(f"Wrote {args.md}")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        _render_csv(rows, args.csv)
        print(f"Wrote {args.csv}")

    # Also print a terse summary to stdout
    by_label: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r["task"])
    print("\nVerdict summary:")
    for label in ("saturated", "dead", "borderline", "discriminative", "no-data"):
        items = by_label.get(label, [])
        if items:
            print(f"  {label:15s} ({len(items):2d}): {', '.join(items)}")


if __name__ == "__main__":
    main()
