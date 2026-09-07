"""Verify Moroco.v2 is dropped from the RoMTEB roster.

Moroco is obsolete for this benchmark. This script does not delete files;
it fails if Moroco.v2 is still registered in benchmark.py / aggregator.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors = []
    bench = (ROOT / "romteb" / "benchmark.py").read_text(encoding="utf-8")
    agg = (ROOT / "scripts" / "aggregate_results.py").read_text(encoding="utf-8")

    if '"Moroco.v2"' in bench and "REUSED_TASK_NAMES" in bench:
        # Allowed only as a comment.
        live = [
            ln
            for ln in bench.splitlines()
            if "Moroco.v2" in ln and not ln.strip().startswith("#")
        ]
        live = [ln for ln in live if "EXCLUDED" not in ln and "dropped" not in ln.lower() and "obsolete" not in ln.lower()]
        # Lines inside REUSED_TASK_NAMES that are string entries.
        registered = [ln for ln in live if '"Moroco.v2"' in ln or "'Moroco.v2'" in ln]
        if registered:
            errors.append("Moroco.v2 still listed in romteb/benchmark.py REUSED_TASK_NAMES")

    if '"Moroco.v2"' in agg:
        in_reused = False
        for ln in agg.splitlines():
            if "REUSED_TASK_NAMES" in ln and "=" in ln:
                in_reused = True
            if in_reused and "}" in ln:
                in_reused = False
            if in_reused and "Moroco.v2" in ln and "EXCLUDED" not in ln:
                errors.append("Moroco.v2 still in scripts/aggregate_results.py REUSED_TASK_NAMES")
                break
        if "Moroco.v2" not in agg.split("EXCLUDED_TASKS", 1)[-1][:1500]:
            errors.append("Moroco.v2 missing from EXCLUDED_TASKS in aggregate_results.py")

    if errors:
        print("Moroco.v2 removal incomplete:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK: Moroco.v2 is not registered; aggregator will skip stale result JSONs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
