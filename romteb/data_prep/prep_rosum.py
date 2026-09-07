"""Investigate and (conditionally) prepare Romanian summarization for RoMTEB.

MTEB's AbsTaskSummarization expects MULTIPLE reference summaries per
article (the metric correlates model-summary similarity to each gold and
takes a max/mean). Most Romanian summarization corpora are single-
reference, which would break the task silently.

Decision policy:
  1. Probe candidate sources.
  2. Report (a) number of summaries per article, (b) total size.
  3. If any source provides >= 2 references per article -> upload and emit
     a TASK_CLASS_TEMPLATE block printed to stdout for manual addition to
     `romteb/tasks/summarization/`.
  4. Otherwise, log a "drop summarization from v1" notice and exit. No
     task class is registered; benchmark.py already excludes summarization.

Sourcing notes:
  - dumitrescustefan/RoSum  (likely single-reference)
  - readerbench/rosum
  - csebuetnlp/xlsum (filter language=='romanian'; single-reference)
"""

from __future__ import annotations

import argparse
from collections import Counter

from romteb._hf_datasets import load_dataset


CANDIDATE_SOURCES = [
    ("dumitrescustefan/RoSum", None),
    ("readerbench/rosum", None),
    ("readerbench/RoSum", None),
    ("csebuetnlp/xlsum", "romanian"),
]


def _probe(repo: str, config: str | None):
    print(f"\n--- Probing {repo}{' / ' + config if config else ''} ---")
    try:
        if config:
            ds = load_dataset(repo, config)
        else:
            ds = load_dataset(repo)
    except Exception as exc:
        print(f"  load failed: {exc}")
        return None
    split = list(ds.keys())[0]
    base = ds[split]
    cols = base.column_names
    print(f"  splits: {list(ds.keys())}  columns: {cols}  rows: {len(base):,}")
    sample = base[0]
    ref_count = _count_references(sample, cols)
    print(f"  sample reference count: {ref_count}")
    return ref_count


def _count_references(sample: dict, cols: list[str]) -> int:
    for key in ("human_summaries", "summaries", "references"):
        if key in cols and isinstance(sample.get(key), list):
            return len(sample[key])
    for key in ("summary", "abstract", "highlights"):
        if key in cols:
            val = sample.get(key)
            if isinstance(val, list):
                return len(val)
            if isinstance(val, str):
                return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    results: dict[str, int] = {}
    for repo, config in CANDIDATE_SOURCES:
        ref_count = _probe(repo, config)
        if ref_count is not None:
            results[f"{repo}{':' + config if config else ''}"] = ref_count

    print("\n=== Decision ===")
    multi_ref = {k: v for k, v in results.items() if v >= 2}
    if multi_ref:
        print(
            "Multi-reference source(s) found: "
            + ", ".join(f"{k} ({v} refs)" for k, v in multi_ref.items())
        )
        print("Manual next step: write a prep_rosum implementation that")
        print("emits `{text: str, human_summaries: list[str]}` and create")
        print("an AbsTaskSummarization class in romteb/tasks/summarization/.")
    else:
        print("No multi-reference corpus found.")
        print("Per plan: DROP summarization from RoMTEB v1.")
        print("Add a row to romteb/tasks/_inventory.md recording this decision.")


if __name__ == "__main__":
    main()
