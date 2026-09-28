"""Flag RoMTEB tasks that do not discriminate between models.

A task earns its slot on the leaderboard only if it *ranks encoders
differently*. We flag four failure modes; a task caught by any of them
is a candidate for demotion (per-task table only, dropped from Overall
/ type-macro / domain means) or full removal.

Failure modes
-------------

1. **BM25 saturation** — a Retrieval / Reranking task where BM25 alone
   already hits an absolute score ≥ `--bm25-abs` AND the best dense
   encoder is within `--bm25-eps` of BM25. Lexical overlap solves it,
   the encoder is barely helping.

2. **Random-baseline collapse** — an MCQ task where every model's
   primary metric is ≤ empirical random baseline (from
   `RERANKING_RANDOM_BASELINE`). Nobody solves it; keeping it in the
   Overall mean would just add noise.

3. **Ceiling collapse (Δ span)** — the top-model / bottom-model gap on
   the primary metric is ≤ `--span-eps` (default 0.02). Every model
   scores essentially the same; the task cannot rank them.

4. **Above-baseline collapse** (classification) — every model scores
   above `--floor` (e.g. 0.90) AND the Δ span is small. The task is
   easy and does not discriminate — the LaRoSeDa case.

Inputs
------

- `--results` (default `results/`): a directory with MTEB-shaped
  outputs `<slug>/<rev>/<Task>.json` for every model in the roster.
- `--bm25-slug` (default `romteb__bm25s-ro`): the model directory used
  as the BM25 baseline row (needed for check 1).

Output
------

Prints a markdown-ready table of flagged tasks with the failure mode
and the raw numbers, plus a JSON dump keyed by task name so the
aggregate scripts can consume it. Nothing is auto-mutated — the
existing `SATURATED_TASKS` / `OVERALL_EXCLUDE_TASKS` frozensets are
edited by hand after a human sanity-checks the report.

Example
-------

    python scripts/audit_task_discrimination.py \\
        --results results \\
        --bm25-slug romteb__bm25s-ro \\
        --json audit_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _iter_result_files(results_dir: Path):
    """Yield (model_slug, task_name, json_dict) for every result file."""
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for rev_dir in model_dir.iterdir():
            if not rev_dir.is_dir():
                continue
            for f in rev_dir.glob("*.json"):
                # Skip the k8 sidecar; discrimination is judged on the primary run.
                if f.name.endswith(".k8.json"):
                    continue
                try:
                    with f.open() as fp:
                        data = json.load(fp)
                except (OSError, ValueError):
                    continue
                yield model_dir.name, f.stem, data


def _primary_score(payload: dict) -> float | None:
    """Return the primary metric from an MTEB result file, or None."""
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return None
    for entries in scores.values():
        if not entries:
            continue
        entry = entries[0] if isinstance(entries, list) else entries
        ms = entry.get("main_score")
        if ms is not None:
            try:
                return float(ms)
            except (TypeError, ValueError):
                continue
    return None


def _collect(results_dir: Path) -> dict[str, dict[str, float]]:
    """Return {task_name: {model_slug: primary_score}}."""
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for slug, task, payload in _iter_result_files(results_dir):
        score = _primary_score(payload)
        if score is None:
            continue
        table[task][slug] = score
    return table


def _task_type(task: str) -> str:
    if task.endswith("PairClassification"):
        return "PairClassification"
    if task.endswith("Classification"):
        return "Classification"
    if task.endswith("Retrieval"):
        return "Retrieval"
    if task.endswith("Reranking"):
        return "Reranking"
    if "BitextMining" in task or task == "Tatoeba":
        return "BitextMining"
    if task == "RoSTS":
        return "STS"
    return "Other"


def _bm25_check(scores: dict[str, float], bm25_slug: str, *, abs_thr: float, eps: float):
    if bm25_slug not in scores:
        return None
    others = {m: s for m, s in scores.items() if m != bm25_slug}
    if not others:
        return None
    bm25 = scores[bm25_slug]
    best_dense = max(others.values())
    if bm25 >= abs_thr and (best_dense - bm25) <= eps:
        return {
            "bm25": round(bm25, 4),
            "best_dense": round(best_dense, 4),
            "gap": round(best_dense - bm25, 4),
        }
    return None


def _random_check(task: str, scores: dict[str, float], baselines: dict):
    entry = baselines.get(task)
    if not entry:
        return None
    key = "accuracy" if "accuracy" in entry else "map_at_1000"
    random_val = float(entry.get(key, 0.0))
    if not scores:
        return None
    best = max(scores.values())
    if best <= random_val + 1e-6:
        return {
            "random_baseline": round(random_val, 4),
            "best_model": round(best, 4),
            "n_models": len(scores),
        }
    return None


def _span_check(scores: dict[str, float], *, eps: float):
    if len(scores) < 2:
        return None
    vals = list(scores.values())
    span = max(vals) - min(vals)
    if span <= eps:
        return {
            "span": round(span, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "n_models": len(vals),
        }
    return None


def _floor_check(scores: dict[str, float], *, floor: float, eps: float):
    if len(scores) < 2:
        return None
    vals = list(scores.values())
    if min(vals) >= floor:
        span = max(vals) - min(vals)
        if span <= eps:
            return {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "span": round(span, 4),
                "floor": floor,
            }
    return None


def _audit(
    per_task: dict[str, dict[str, float]],
    *,
    bm25_slug: str,
    bm25_abs: float,
    bm25_eps: float,
    span_eps: float,
    floor_min: float,
    floor_eps: float,
):
    try:
        from romteb.eval_config import RERANKING_RANDOM_BASELINE
    except ImportError:
        RERANKING_RANDOM_BASELINE = {}

    findings = []
    for task in sorted(per_task):
        scores = per_task[task]
        ttype = _task_type(task)
        reasons = {}

        if ttype in {"Retrieval", "Reranking"}:
            hit = _bm25_check(scores, bm25_slug, abs_thr=bm25_abs, eps=bm25_eps)
            if hit:
                reasons["bm25_saturated"] = hit

        hit = _random_check(task, scores, RERANKING_RANDOM_BASELINE)
        if hit:
            reasons["random_collapse"] = hit

        hit = _span_check(scores, eps=span_eps)
        if hit:
            reasons["span_collapse"] = hit

        if ttype == "Classification":
            hit = _floor_check(scores, floor=floor_min, eps=floor_eps)
            if hit:
                reasons["above_floor_collapse"] = hit

        if reasons:
            findings.append({
                "task": task,
                "type": ttype,
                "n_models": len(scores),
                "reasons": reasons,
            })
    return findings


def _print_table(findings: list[dict]) -> None:
    if not findings:
        print("No tasks flagged.")
        return
    print("| Task | Type | Reasons | Detail |")
    print("|---|---|---|---|")
    for f in findings:
        reasons = ", ".join(f["reasons"])
        detail_bits = []
        for k, v in f["reasons"].items():
            detail_bits.append(f"{k}: {v}")
        print(f"| {f['task']} | {f['type']} | {reasons} | {' \\ '.join(detail_bits)} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument(
        "--bm25-slug",
        default="romteb__bm25s-ro",
        help="Model directory name for the BM25 baseline row",
    )
    parser.add_argument("--bm25-abs", type=float, default=0.85)
    parser.add_argument("--bm25-eps", type=float, default=0.05)
    parser.add_argument("--span-eps", type=float, default=0.02)
    parser.add_argument("--floor", dest="floor_min", type=float, default=0.90)
    parser.add_argument("--floor-eps", type=float, default=0.03)
    parser.add_argument("--json", help="Write the report as JSON here")
    args = parser.parse_args(argv)

    results = Path(args.results)
    if not results.is_dir():
        raise SystemExit(f"[audit] results dir not found: {results}")

    per_task = _collect(results)
    findings = _audit(
        per_task,
        bm25_slug=args.bm25_slug,
        bm25_abs=args.bm25_abs,
        bm25_eps=args.bm25_eps,
        span_eps=args.span_eps,
        floor_min=args.floor_min,
        floor_eps=args.floor_eps,
    )
    _print_table(findings)

    if args.json:
        Path(args.json).write_text(json.dumps(findings, indent=2, ensure_ascii=False))
        print(f"\n[audit] wrote {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
