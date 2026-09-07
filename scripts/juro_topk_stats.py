"""Measure JuRo |top_ranked| and optionally fix the random baseline.

USAGE:
    ./apptainer-exec-romteb.sh scripts/juro_topk_stats.py
    ./apptainer-exec-romteb.sh scripts/juro_topk_stats.py --update_config
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def _harmonic(n: int) -> float:
    return sum(1.0 / i for i in range(1, n + 1))


def _load_pool_sizes() -> list[int]:
    from romteb._hf_datasets import load_dataset

    ds = load_dataset("alina0195/romteb-juro-legal-reranking", "top_ranked", split="test")
    col = "corpus-ids" if "corpus-ids" in ds.column_names else "corpus_ids"
    return [len(row[col]) for row in ds]


def _patch_eval_config(k: int, acc: float, map_val: float) -> None:
    path = ROOT / "romteb" / "eval_config.py"
    text = path.read_text(encoding="utf-8")
    pattern = r'"JuRoLegalExamReranking": \{"k": \d+, "accuracy": [0-9.]+, "map_at_1000": [0-9.]+\}'
    repl = (
        f'"JuRoLegalExamReranking": {{"k": {k}, "accuracy": {acc:.3f}, '
        f'"map_at_1000": {map_val:.3f}}}'
    )
    new, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        raise SystemExit(f"could not patch JuRo baseline in {path}")
    path.write_text(new, encoding="utf-8")
    print(f"Updated {path.relative_to(ROOT)}: k={k} acc={acc:.3f} MAP={map_val:.3f}")


def _patch_protocol(k: int, acc: float) -> None:
    path = ROOT / "docs" / "evaluation_protocol.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    old = (
        "| `JuRoLegalExamReranking` | întrebare juridică | "
        "A–E (corpus ≈ 3 opțiuni/q; `k=5` de verificat) | "
        "varianta corectă | accuracy@1 | 0.200 |"
    )
    new = (
        f"| `JuRoLegalExamReranking` | întrebare juridică | "
        f"pool per-întrebare (mean k={k}) | "
        f"varianta corectă | accuracy@1 | {acc:.3f} |"
    )
    if old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"Updated {path.relative_to(ROOT)}")
    else:
        print(f"[warn] protocol table row not found in {path}; skip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/juro_topk_stats.txt")
    parser.add_argument(
        "--update_config",
        action="store_true",
        help="Rewrite RERANKING_RANDOM_BASELINE for JuRo from measured k.",
    )
    args = parser.parse_args()

    lens = _load_pool_sizes()
    mean_k = mean(lens)
    e_acc = mean(1.0 / k for k in lens)
    e_map = mean(_harmonic(k) / k for k in lens)
    k_int = int(round(mean_k))
    lines = [
        f"n_queries={len(lens)}",
        f"mean_k={mean_k:.3f}",
        f"min={min(lens)}",
        f"max={max(lens)}",
        f"mode_like_round={k_int}",
        f"E[acc@1] if uniform 1/k = {e_acc:.3f}",
        f"E[MAP] if uniform rank = {e_map:.3f}",
        f"H_{k_int}/k_int = {_harmonic(k_int) / k_int:.3f}",
    ]
    report = "\n".join(lines) + "\n"
    print(report, end="")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")

    if args.update_config:
        _patch_eval_config(k_int, e_acc, e_map)
        _patch_protocol(k_int, e_acc)


if __name__ == "__main__":
    main()
