"""Statistically explicit score extraction and hierarchical aggregation.

Atomic cell
    (embedder, task): mean of the MTEB primary metric on the eval split.

    Classification: mean ± sample std over ``n_experiments`` logistic probes.
    Bitext / multilingual tasks: mean ± sample std over HF subsets (language
    pairs), never the first pair only.
    Retrieval / STS / PairClassification: a single split score (no seed).
    Clustering: mean V-measure; std is MTEB's bootstrap ``v_measure_std``.

Hierarchy (never mix metrics; never double-count a dataset inside one slice)
    task mean
      -> dataset mean  (unweighted mean of that source's tasks *of the same type*)
      -> domain × type (unweighted mean of datasets)
      -> type mean     (unweighted mean of datasets in that MTEB category)
      -> Overall official = category Borda then mean of category ranks
                       plus type-macro / task-macro as side columns.

Missing cells are omitted, not imputed. Confirmed contaminated (model, task)
pairs are dropped from every mean that would include them. ``unknown``
contamination is kept and flagged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

SPLIT_PRIORITY = ("test", "validation", "dev")


@dataclass(frozen=True)
class ScoreStats:
    mean: float
    std: float | None
    n: int
    split: str
    f1: float | None = None
    n_subsets: int = 1
    accuracy: float | None = None
    map_at_1000: float | None = None


def _finite(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _split_entries(result: dict) -> tuple[str, list[dict]] | None:
    scores = result.get("scores")
    if not isinstance(scores, dict):
        return None
    for split in SPLIT_PRIORITY:
        raw = scores.get(split)
        if not raw:
            continue
        if isinstance(raw, dict):
            entries = [raw]
        elif isinstance(raw, list):
            entries = [e for e in raw if isinstance(e, dict)]
        else:
            continue
        if entries:
            return split, entries
    return None


def _experiment_values(entry: dict, key: str) -> list[float]:
    raw = entry.get("scores_per_experiment")
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for exp in raw:
        if not isinstance(exp, dict):
            continue
        v = _finite(exp.get(key))
        if v is None:
            v = _finite(exp.get("main_score"))
        if v is not None:
            out.append(v)
    return out


def extract_score_stats(result: dict, *, metric_key: str | None = None) -> ScoreStats | None:
    """Return the statistically correct main score for one result JSON.

    ``metric_key`` defaults to the entry's ``main_score`` (and, for classification
    seeds, ``accuracy``). All HF subsets in the chosen split are averaged.
    Clustering bootstrap ``v_measure_std`` is used as std when present.
    """
    parsed = _split_entries(result)
    if parsed is None:
        v = _finite(result.get("main_score"))
        if v is None:
            return None
        return ScoreStats(mean=v, std=None, n=1, split="unknown")

    split, entries = parsed
    seed_key = metric_key or "accuracy"
    subset_means: list[float] = []
    seed_std: float | None = None
    seed_n = 0
    f1s: list[float] = []
    accuracies: list[float] = []
    maps: list[float] = []
    cluster_stds: list[float] = []

    for entry in entries:
        seeds = _experiment_values(entry, seed_key)
        if len(seeds) >= 2:
            subset_means.append(mean(seeds))
            seed_std = stdev(seeds)
            seed_n = len(seeds)
        else:
            v = _finite(entry.get(metric_key)) if metric_key else None
            if v is None:
                v = _finite(entry.get("main_score"))
            if v is None:
                continue
            subset_means.append(v)
        f1 = _finite(entry.get("f1"))
        if f1 is not None:
            f1s.append(f1)
        acc = _finite(entry.get("accuracy"))
        if acc is not None:
            accuracies.append(acc)
        mp = _finite(entry.get("map_at_1000"))
        if mp is not None:
            maps.append(mp)
        vst = _finite(entry.get("v_measure_std"))
        if vst is not None:
            cluster_stds.append(vst)

    if not subset_means:
        return None

    n_subsets = len(subset_means)
    grand = mean(subset_means)
    if n_subsets >= 2:
        std = stdev(subset_means)
        n = n_subsets
    elif seed_n >= 2:
        std = seed_std
        n = seed_n
    elif cluster_stds:
        std = mean(cluster_stds)
        n = 1
    else:
        std = None
        n = 1
    return ScoreStats(
        mean=grand,
        std=std,
        n=n,
        split=split,
        f1=mean(f1s) if f1s else None,
        n_subsets=n_subsets,
        accuracy=mean(accuracies) if accuracies else None,
        map_at_1000=mean(maps) if maps else None,
    )


def format_score(stats: ScoreStats | None, *, with_std: bool = True) -> str:
    if stats is None:
        return "-"
    if with_std and stats.std is not None:
        return f"{stats.mean:.4f} ± {stats.std:.4f}"
    return f"{stats.mean:.4f}"


def stderr(stats: ScoreStats) -> float | None:
    if stats.std is None or stats.n < 2:
        return None
    return stats.std / (stats.n ** 0.5)


def sample_mean_std(values: Sequence[float]) -> tuple[float, float | None, int]:
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        raise ValueError("empty")
    if n == 1:
        return xs[0], None, 1
    return mean(xs), stdev(xs), n


def average_ranks(
    model_to_score: dict[str, float],
    *,
    higher_is_better: bool = True,
) -> dict[str, float]:
    """Competition ranks with average ties; rank 1 is best."""
    if not model_to_score:
        return {}
    grouped: dict[float, list[str]] = defaultdict(list)
    for model, score in model_to_score.items():
        grouped[score].append(model)
    ranks: dict[str, float] = {}
    current = 1
    ordered = sorted(grouped, reverse=higher_is_better)
    for score in ordered:
        models = grouped[score]
        avg_rank = current + (len(models) - 1) / 2.0
        for m in models:
            ranks[m] = avg_rank
        current += len(models)
    return ranks


def borda_scores(
    model_to_score: dict[str, float],
    *,
    higher_is_better: bool = True,
) -> dict[str, float]:
    """Borda points: ``n - rank`` with average ties. Higher is better."""
    ranks = average_ranks(model_to_score, higher_is_better=higher_is_better)
    n = len(model_to_score)
    return {model: n - rank for model, rank in ranks.items()}


def mean_rank_across_tasks(
    # model -> task -> score
    scores: dict[str, dict[str, float]],
    task_names: Iterable[str],
) -> dict[str, tuple[float, int]]:
    """Mean rank per model over tasks; missing cells skip that (model, task)."""
    rank_lists: dict[str, list[float]] = defaultdict(list)
    for task in task_names:
        present = {
            model: tasks[task]
            for model, tasks in scores.items()
            if task in tasks
        }
        if len(present) < 2:
            continue
        for model, rank in average_ranks(present).items():
            rank_lists[model].append(rank)
    out: dict[str, tuple[float, int]] = {}
    for model, ranks in rank_lists.items():
        out[model] = (mean(ranks), len(ranks))
    return out


def category_borda_then_mean_rank(
    # model -> (category, dataset) -> score
    dataset_scores: dict[str, dict[tuple[str, str], float]],
    categories: Iterable[str],
) -> dict[str, tuple[float, int]]:
    """Official Overall: Borda inside each category, then mean of category ranks.

    Each dataset in a category is a voter (after source collapse). Borda
    points are summed per model inside the category; those sums are ranked;
    the unweighted mean of category ranks is the Overall Borda rank
    (1 = best). Missing (model, dataset) cells skip that voter.
    """
    models = list(dataset_scores)
    category_ranks: dict[str, list[float]] = defaultdict(list)
    for category in categories:
        borda_totals: dict[str, float] = defaultdict(float)
        n_voters = 0
        datasets = sorted(
            {
                dataset
                for tasks in dataset_scores.values()
                for (cat, dataset) in tasks
                if cat == category
            }
        )
        for dataset in datasets:
            present = {
                model: scores[(category, dataset)]
                for model, scores in dataset_scores.items()
                if (category, dataset) in scores
            }
            if len(present) < 2:
                continue
            n_voters += 1
            for model, pts in borda_scores(present).items():
                borda_totals[model] += pts
        if n_voters == 0 or len(borda_totals) < 2:
            continue
        for model, rank in average_ranks(dict(borda_totals)).items():
            category_ranks[model].append(rank)
    out: dict[str, tuple[float, int]] = {}
    for model in models:
        ranks = category_ranks.get(model, [])
        if ranks:
            out[model] = (mean(ranks), len(ranks))
    return out
