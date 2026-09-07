"""Smoke-test RoMTEB end-to-end with a tiny task subset.

Phase 8 sanity check: verifies that the benchmark assembles, MTEB v2
accepts our task classes, the runner produces a JSON file, and the
aggregator can parse it.

Usage:
    ./apptainer-exec-romteb.sh romteb/scripts/smoke_test.py
    ./apptainer-exec-romteb.sh romteb/scripts/smoke_test.py --model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

from romteb.benchmark import ROMTEB_CUSTOM, ROMTEB_REUSED, ROMTEB_TASKS


SMOKE_TASK_NAMES = ["RoSTS", "RoDTALLawsRetrieval"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--output", default="results/_smoke")
    args = parser.parse_args()

    print(f"mteb={mteb.__version__}")
    print(f"ROMTEB_TASKS: {len(ROMTEB_TASKS)} total ({len(ROMTEB_REUSED)} reused, {len(ROMTEB_CUSTOM)} custom)")
    for t in ROMTEB_TASKS:
        print(f"  - {t.metadata.name:40s} {t.metadata.type}")

    wanted = set(SMOKE_TASK_NAMES)
    task_list = [t for t in ROMTEB_TASKS if t.metadata.name in wanted]
    if not task_list:
        sys.exit(f"None of {SMOKE_TASK_NAMES} present in benchmark. Phase 1/6 incomplete.")

    print(f"\nSmoke-running {[t.metadata.name for t in task_list]} on {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    evaluation = mteb.MTEB(tasks=task_list)
    evaluation.run(model, output_folder=args.output, overwrite_results=True)

    out_path = Path(args.output)
    jsons = list(out_path.rglob("*.json"))
    print(f"\nGenerated {len(jsons)} JSON files under {out_path}:")
    for p in jsons:
        print(f"  - {p}")

    if not jsons:
        sys.exit("No result files produced - check the runner output above.")
    print("\nSmoke test OK. Now run scripts/aggregate_results.py to verify parsing.")


if __name__ == "__main__":
    main()
