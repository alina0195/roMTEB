"""CLI entry point for platform workers: ``romteb-eval``.

Examples:
    romteb-eval --model intfloat/multilingual-e5-large --preset smoke --output /out
    romteb-eval --model /uploads/job-42 --preset full --output /out --summary /out/summary.json
    romteb-eval --list-tasks
    romteb-eval --summary-only --model intfloat/multilingual-e5-large --output results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from romteb.presets import PRESET_NAMES, describe_presets, resolve_task_names
from romteb.summary import write_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="romteb-eval",
        description="Run the RoMTEB evaluation protocol and write summary.json.",
    )
    parser.add_argument(
        "--model",
        help="HF model id, local encoder directory, or romteb/bm25s-ro",
    )
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default="full",
        help="smoke=RoSTS, core=Classification+Pair+STS, full=v1 protocol (default)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Restrict to these task names (overrides --preset)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Path for summary.json (default: <output>/summary.json)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not run eval; rebuild summary.json from existing JSON under --output",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print protocol presets/tasks as JSON and exit (no GPU)",
    )
    parser.add_argument("--job-id", default=None, help="Copied into summary.json")
    parser.add_argument("--overwrite_results", action="store_true")
    parser.add_argument(
        "--loader",
        choices=["auto", "st", "mteb"],
        default="auto",
        help=(
            "Model loader: 'st' = SentenceTransformer + MODEL_PROMPTS, "
            "'mteb' = mteb.get_model, 'auto' = mteb wrapper when declared else ST"
        ),
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash-attn (also ROMTEB_DISABLE_FLASH_ATTN=1).",
    )
    parser.add_argument(
        "--skip_k8",
        action="store_true",
        help="Do not run the samples_per_label=8 classification sidecar.",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow dense models on CPU (debug only).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Accepted for roster compatibility; default is already True.",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Refuse custom modeling code (recommended for untrusted uploads).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Encode batch size (MTEB default 32). Use 1–4 for 8B+/12B OOM retries.",
    )
    return parser


def _list_tasks() -> int:
    print(json.dumps(describe_presets(), indent=2, ensure_ascii=False))
    return 0


def aggregate_main() -> None:
    """Entry point: ``romteb-aggregate``."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "aggregate_results.py"
    if not script.is_file():
        raise SystemExit(f"[romteb] aggregator not found at {script}")
    import runpy

    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.list_tasks:
        return _list_tasks()

    if not args.model and not args.summary_only:
        _parser().error("--model is required unless --list-tasks or --summary-only")

    summary_path = args.summary or str(Path(args.output) / "summary.json")

    if args.summary_only:
        write_summary(
            args.output,
            summary_path,
            model=args.model,
            preset=args.preset,
            job_id=args.job_id,
        )
        return 0

    from romteb.benchmark import ROMTEB_TASKS
    from romteb.run_benchmark import _flash_attn_disabled, run_romteb

    task_names = resolve_task_names(
        preset=args.preset,
        tasks=args.tasks,
        catalog=ROMTEB_TASKS,
    )
    result = run_romteb(
        args.model,
        output_dir=args.output,
        tasks=task_names,
        overwrite_results=args.overwrite_results,
        loader=args.loader,
        no_flash_attn=args.no_flash_attn or _flash_attn_disabled(),
        skip_k8=args.skip_k8,
        allow_cpu=args.allow_cpu,
        batch_size=args.batch_size,
        trust_remote_code=not args.no_trust_remote_code,
    )
    failed = list(result.get("failed_tasks") or [])
    write_summary(
        args.output,
        summary_path,
        model=args.model,
        failed_tasks=failed,
        preset=args.preset if not args.tasks else None,
        job_id=args.job_id,
    )
    if failed:
        print(f"[romteb] {len(failed)} task(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
