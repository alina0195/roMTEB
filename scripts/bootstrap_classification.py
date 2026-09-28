"""Bootstrap CIs on Classification / PairClassification per-example predictions.

Companion to ``scripts/bootstrap_ir.py``. Under the full-train protocol
``n_experiments = 1``, so the historical 10-draw sample std is gone; the
uncertainty on accuracy / macro-F1 / average precision now comes from
resampling the test-set predictions.

Predictions must be saved during the run. That happens automatically when
``romteb.run_benchmark._run_task`` is called against a recent mteb version;
per-example prediction files land under

    <results_dir>/predictions/<model-slug>/<TaskName>_predictions.json

with the mteb schema ``{mteb_model_meta, <subset>: {<split>: [
{"true": <label>, "pred": <label>, ...}, ...]}}``. Predictions written by
``RoNLIClassification._save_task_predictions`` follow the same shape.

USAGE:
    ./apptainer-exec-romteb.sh scripts/bootstrap_classification.py \\
        --results_dir results
    ./apptainer-exec-romteb.sh scripts/bootstrap_classification.py \\
        --results_dir results --task RoNLIClassification
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from romteb.eval_config import (
    LEADERBOARD_EXCLUDE,
    REPORT_MACRO_F1,
    metric_of,
)

N_BOOT = 1000
RNG = np.random.default_rng(42)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_prediction_files(pred_root: Path) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    if not pred_root.exists():
        return out
    for path in pred_root.rglob("*_predictions.json"):
        task = path.name[: -len("_predictions.json")]
        try:
            slug = path.relative_to(pred_root).parts[0]
        except ValueError:
            slug = path.parent.name
        out.append((slug, task, path))
    return out


def _extract_pairs(blob: dict) -> list[tuple[object, object]]:
    """Return [(true, pred), ...] from an mteb classification predictions file.

    Accepts the standard shape (subset -> split -> list of rows) or a flat list.
    A row may key its labels as {"true","pred"}, {"y_true","y_pred"},
    {"label","prediction"}, or {"gold","predicted"}.
    """
    key_pairs = (("true", "pred"), ("y_true", "y_pred"),
                 ("label", "prediction"), ("gold", "predicted"))

    def _row_pair(row: dict) -> tuple[object, object] | None:
        for tk, pk in key_pairs:
            if tk in row and pk in row:
                return row[tk], row[pk]
        return None

    def _walk(node) -> list[tuple[object, object]]:
        rows: list[tuple[object, object]] = []
        if isinstance(node, list):
            for x in node:
                if isinstance(x, dict):
                    pair = _row_pair(x)
                    if pair is not None:
                        rows.append(pair)
        elif isinstance(node, dict):
            for k, v in node.items():
                if k == "mteb_model_meta":
                    continue
                rows.extend(_walk(v))
        return rows

    return _walk(blob)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if y_true.size else float("nan")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in labels:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp == 0 and (fp == 0 or fn == 0):
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int,
) -> tuple[float, float, float]:
    n = y_true.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = metric_fn(y_true, y_pred)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        draws[i] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    parser.add_argument("--task", default=None)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    root = Path(args.results_dir)
    pred_root = root / "predictions"
    out_csv = Path(args.out_csv) if args.out_csv else root / "romteb_classification_bootstrap.csv"

    files = _iter_prediction_files(pred_root)
    if not files:
        print(
            f"[bootstrap-cls] no prediction files under {pred_root}. "
            "Re-run classification with a recent mteb so per-example "
            "predictions are saved."
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text(
            "model,task,metric,mean,ci_lo,ci_hi,n\n", encoding="utf-8"
        )
        return

    drop_slugs = {m.replace("/", "__") for m in LEADERBOARD_EXCLUDE}
    rows: list[dict] = []
    grouped: dict[tuple[str, str], list[tuple[object, object]]] = defaultdict(list)
    for slug, task, path in files:
        if slug in drop_slugs or (args.task and task != args.task):
            continue
        try:
            grouped[(slug, task)].extend(_extract_pairs(_load_json(path)))
        except Exception as exc:
            print(f"[bootstrap-cls] skip {slug}/{task}: {exc}")

    print("per-example bootstrap 95% CI")
    for (slug, task), pairs in sorted(grouped.items()):
        if not pairs:
            continue
        try:
            y_true = np.asarray([p[0] for p in pairs])
            y_pred = np.asarray([p[1] for p in pairs])
        except Exception as exc:
            print(f"[bootstrap-cls] skip {slug}/{task}: labels not array-able ({exc})")
            continue

        metrics = ["accuracy"]
        # PairClassification's primary is average precision; without scores we
        # cannot bootstrap AP, so we still report accuracy for those files.
        if task in REPORT_MACRO_F1 or metric_of("Classification", task) == "f1":
            metrics.append("macro_f1")

        for metric in metrics:
            fn = _accuracy if metric == "accuracy" else _macro_f1
            mu, lo, hi = _bootstrap_ci(y_true, y_pred, fn, args.n_boot)
            model = slug.replace("__", "/")
            print(
                f"  {model}  {task}  {metric}: "
                f"{mu:.4f}  [{lo:.4f}, {hi:.4f}]  n={y_true.size}"
            )
            rows.append({
                "model": model,
                "task": task,
                "metric": metric,
                "mean": mu,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": int(y_true.size),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "task", "metric", "mean", "ci_lo", "ci_hi", "n"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
