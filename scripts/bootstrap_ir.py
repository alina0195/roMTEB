"""Bootstrap CIs and paired permutation tests on IR per-query scores.

Requires predictions saved by ``run_benchmark`` under
``<results_dir>/predictions/<model-slug>/<TaskName>_predictions.json``.

USAGE:
    ./apptainer-exec-romteb.sh scripts/bootstrap_ir.py --results_dir results
    ./apptainer-exec-romteb.sh scripts/bootstrap_ir.py \\
        --results_dir results --pair model_a model_b --task RoDTALLawsRetrieval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from romteb.eval_config import LEADERBOARD_EXCLUDE, metric_of
from romteb.scoring import _finite

try:
    import pytrec_eval
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pytrec_eval is required for bootstrap_ir.py") from exc

N_BOOT = 1000
N_PERM = 5000
RNG = np.random.default_rng(42)

METRIC_TO_PYTREC = {
    "ndcg_at_10": "ndcg_cut_10",
    "map_at_1000": "map_cut_1000",
    "accuracy": "recall_1",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_predictions(blob: dict) -> dict[str, dict[str, float]]:
    """MTEB prediction file: {mteb_model_meta, subset: {split: {qid: {doc: score}}}}."""
    for key, val in blob.items():
        if key == "mteb_model_meta" or not isinstance(val, dict):
            continue
        for _split, ranking in val.items():
            if isinstance(ranking, dict) and ranking and isinstance(
                next(iter(ranking.values()), None), dict
            ):
                sample = next(iter(ranking.values()))
                if sample and isinstance(next(iter(sample.values()), None), (int, float)):
                    return ranking
    raise ValueError("could not find qid -> {doc: score} rankings in prediction file")


def _qrels_from_task(task_name: str) -> dict[str, dict[str, int]]:
    from romteb.benchmark import ROMTEB_TASKS

    task = next((t for t in ROMTEB_TASKS if t.metadata.name == task_name), None)
    if task is None:
        raise SystemExit(f"task not in RoMTEB: {task_name}")
    task.load_data()
    split = (task.metadata.eval_splits or ["test"])[0]
    dataset = task.dataset
    blob = None
    if split in dataset and isinstance(dataset[split], dict) and "relevant_docs" in dataset[split]:
        blob = dataset[split]
    else:
        for _subset, splits in dataset.items():
            if isinstance(splits, dict) and split in splits:
                cand = splits[split]
                if isinstance(cand, dict) and "relevant_docs" in cand:
                    blob = cand
                    break
    if blob is None:
        raise SystemExit(f"no qrels for {task_name}")
    rel = blob["relevant_docs"]
    out: dict[str, dict[str, int]] = {}
    for qid, docs in rel.items():
        if isinstance(docs, dict):
            out[str(qid)] = {str(d): int(s) for d, s in docs.items()}
        else:
            out[str(qid)] = {str(d): 1 for d in docs}
    return out


def _measure(task_name: str, task_type: str) -> tuple[str, str]:
    key = metric_of(task_type, task_name)
    pytrec = METRIC_TO_PYTREC.get(key, "ndcg_cut_10")
    return key, pytrec


def per_query_scores(
    rankings: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    pytrec_measure: str,
) -> dict[str, float]:
    measures = {pytrec_measure.replace("_", ".", 1) if "cut" in pytrec_measure or pytrec_measure.startswith("recall") else pytrec_measure}
    # pytrec wants "ndcg_cut.10", "map_cut.1000", "recall.1"
    if pytrec_measure == "ndcg_cut_10":
        measures = {"ndcg_cut.10"}
        out_key = "ndcg_cut_10"
    elif pytrec_measure == "map_cut_1000":
        measures = {"map_cut.1000"}
        out_key = "map_cut_1000"
    elif pytrec_measure == "recall_1":
        measures = {"recall.1"}
        out_key = "recall_1"
    else:
        measures = {pytrec_measure}
        out_key = pytrec_measure
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    scored = evaluator.evaluate(rankings)
    out = {}
    for qid, vals in scored.items():
        v = _finite(vals.get(out_key))
        if v is None:
            for k, raw in vals.items():
                v = _finite(raw)
                if v is not None:
                    break
        if v is not None:
            out[qid] = v
    return out


def bootstrap_ci(values: list[float], n_boot: int = N_BOOT) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        sample = arr[RNG.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def permutation_pvalue(a: list[float], b: list[float], n_perm: int = N_PERM) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size or x.size == 0:
        return float("nan")
    observed = abs(x.mean() - y.mean())
    diff = x - y
    count = 0
    for _ in range(n_perm):
        signs = RNG.choice(np.array([-1.0, 1.0]), size=diff.size)
        if abs((signs * diff).mean()) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def _iter_prediction_files(pred_root: Path) -> list[tuple[str, str, Path]]:
    out = []
    if not pred_root.exists():
        return out
    for path in pred_root.rglob("*_predictions.json"):
        task = path.name[: -len("_predictions.json")]
        # predictions/<model-slug>/Task_predictions.json
        try:
            slug = path.relative_to(pred_root).parts[0]
        except ValueError:
            slug = path.parent.name
        out.append((slug, task, path))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    parser.add_argument("--n_perm", type=int, default=N_PERM)
    parser.add_argument("--pair", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    parser.add_argument("--task", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_pair_csv", default=None)
    args = parser.parse_args()

    root = Path(args.results_dir)
    pred_root = root / "predictions"
    out_csv = Path(args.out_csv) if args.out_csv else root / "romteb_ir_bootstrap.csv"
    out_pair = Path(args.out_pair_csv) if args.out_pair_csv else root / "romteb_ir_pairwise.csv"
    files = _iter_prediction_files(pred_root)
    if not files:
        print(
            f"[bootstrap] no prediction files under {pred_root}. "
            "Skip CI (re-run retrieval/reranking to populate predictions/)."
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("model,task,mean,ci_lo,ci_hi,n\n", encoding="utf-8")
        return

    qrels_cache: dict[str, dict] = {}
    scores: dict[tuple[str, str], dict[str, float]] = {}
    drop_slugs = {m.replace("/", "__") for m in LEADERBOARD_EXCLUDE}
    for slug, task, path in files:
        if slug in drop_slugs:
            continue
        if args.task and task != args.task:
            continue
        try:
            rankings = _flatten_predictions(_load_json(path))
            if task not in qrels_cache:
                qrels_cache[task] = _qrels_from_task(task)
            task_type = "Reranking" if "Reranking" in task else "Retrieval"
            _key, pytrec_m = _measure(task, task_type)
            scores[(slug, task)] = per_query_scores(
                rankings, qrels_cache[task], pytrec_m
            )
        except Exception as exc:
            print(f"[bootstrap] skip {slug}/{task}: {exc}")

    print("per-query bootstrap 95% CI")
    boot_rows: list[dict] = []
    for (slug, task), per_q in sorted(scores.items()):
        mu, lo, hi = bootstrap_ci(list(per_q.values()), n_boot=args.n_boot)
        model = slug.replace("__", "/")
        print(f"  {model}  {task}: {mu:.4f}  [{lo:.4f}, {hi:.4f}]  n={len(per_q)}")
        boot_rows.append(
            {
                "model": model,
                "task": task,
                "mean": mu,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": len(per_q),
            }
        )
    write_boot = args.out_csv is not None or not args.task
    if write_boot:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["model", "task", "mean", "ci_lo", "ci_hi", "n"])
            writer.writeheader()
            writer.writerows(boot_rows)
        print(f"Wrote {out_csv} ({len(boot_rows)} rows)")
    else:
        print("[bootstrap] --task set: not overwriting full bootstrap CSV")

    if args.pair:
        a, b = args.pair
        a_slug, b_slug = a.replace("/", "__"), b.replace("/", "__")
        tasks = sorted({t for (s, t) in scores if s in {a_slug, b_slug}})
        if args.task:
            tasks = [args.task]
        print(f"\npaired permutation tests ({a} vs {b})")
        pair_rows: list[dict] = []
        for task in tasks:
            sa = scores.get((a_slug, task))
            sb = scores.get((b_slug, task))
            if not sa or not sb:
                continue
            qids = sorted(set(sa) & set(sb))
            p = permutation_pvalue([sa[q] for q in qids], [sb[q] for q in qids], args.n_perm)
            da = float(np.mean([sa[q] for q in qids]))
            db = float(np.mean([sb[q] for q in qids]))
            print(f"  {task}: Δ={da - db:+.4f}  p={p:.4f}  n={len(qids)}")
            pair_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "task": task,
                    "mean_a": da,
                    "mean_b": db,
                    "delta_a_minus_b": da - db,
                    "p_value": p,
                    "n": len(qids),
                }
            )
        if pair_rows:
            out_pair.parent.mkdir(parents=True, exist_ok=True)
            write_header = not out_pair.exists()
            with out_pair.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "model_a",
                        "model_b",
                        "task",
                        "mean_a",
                        "mean_b",
                        "delta_a_minus_b",
                        "p_value",
                        "n",
                    ],
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(pair_rows)
            print(f"Wrote {out_pair} ({len(pair_rows)} rows)")


if __name__ == "__main__":
    main()
