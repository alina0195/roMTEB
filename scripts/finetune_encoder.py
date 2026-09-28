"""Fine-tune a sentence encoder on a RoMTEB train split.

Frozen vs fine-tuned encoder is one of the open questions in the RoMTEB
protocol: for STS and PairClassification we want to report both.

Frozen (default RoMTEB path):
    romteb-eval --model <hf-id> --tasks RoSTS RoNLIPairClassification ...

Fine-tuned (this script):
    python scripts/finetune_encoder.py sts \
        --model intfloat/multilingual-e5-large --output ckpt/e5-ro-sts
    romteb-eval --model ckpt/e5-ro-sts --tasks RoSTS ...

    python scripts/finetune_encoder.py pair \
        --model intfloat/multilingual-e5-large --output ckpt/e5-ro-ronli
    romteb-eval --model ckpt/e5-ro-ronli --tasks RoNLIPairClassification ...

The eval CLI is unchanged: it just loads the fine-tuned checkpoint from
disk. Every leaderboard row for the fine-tuned variant should carry a
``_ft`` suffix in the model directory so the frozen and fine-tuned
scores never end up in the same aggregate.

STS training
    Cosine-similarity regression on the RoSTS train split. Gold scores
    normalised from [0, 5] → [0, 1] as sentence-transformers expects.

Pair training
    RoNLI Entailment vs Contrastive with ``ContrastiveLoss`` (metric
    learning): positives (Entailment) are pulled together, negatives
    (Contrastive) pushed apart with a margin. Consequence + Unrelated
    are excluded to match the eval task's XNLI-style framing.

This script is intentionally minimal: no LR search, no early stopping.
It exists to unblock the frozen-vs-fine-tuned comparison, not to tune
a SOTA checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


def _lazy_imports():
    """Import heavy deps only when a subcommand actually runs."""
    try:
        from datasets import load_dataset
        from sentence_transformers import (
            InputExample,
            SentenceTransformer,
            losses,
        )
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "[finetune] missing dependency: install "
            "`sentence-transformers`, `datasets`, `torch`"
        ) from exc
    return {
        "load_dataset": load_dataset,
        "InputExample": InputExample,
        "SentenceTransformer": SentenceTransformer,
        "losses": losses,
        "DataLoader": DataLoader,
    }


def _read_revision(hf_path: str) -> str | None:
    """Return the pinned revision for a Hub dataset, or None if unpinned."""
    try:
        from romteb.data_prep._common import read_revision
    except ImportError:
        return None
    try:
        return read_revision(hf_path)
    except Exception:  # pragma: no cover - falls back to `main`
        return None


def _rosts_examples(row, *, InputExample):
    # RoSTS ships scores in [0, 5]; sentence-transformers CosineSimilarityLoss
    # expects [0, 1].
    return InputExample(
        texts=[row["sentence1"], row["sentence2"]],
        label=float(row["score"]) / 5.0,
    )


def _ronli_examples(row, *, InputExample):
    # 0 Contrastive, 1 Entailment (see romteb/tasks/pair_classification/ronli.py).
    return InputExample(
        texts=[row["sentence1"], row["sentence2"]],
        label=1.0 if int(row["label"]) == 1 else 0.0,
    )


def _load_split(hf_path: str, split: str, revision: str | None):
    deps = _lazy_imports()
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return deps["load_dataset"](hf_path, **kwargs)


def _train(
    *,
    model_id: str,
    output_dir: Path,
    examples: Iterable,
    loss_ctor,
    epochs: int,
    batch_size: int,
    warmup_ratio: float,
) -> None:
    deps = _lazy_imports()
    SentenceTransformer = deps["SentenceTransformer"]
    DataLoader = deps["DataLoader"]

    model = SentenceTransformer(model_id)
    loader = DataLoader(list(examples), shuffle=True, batch_size=batch_size)
    loss = loss_ctor(model=model)
    total_steps = len(loader) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    print(f"[finetune] wrote checkpoint to {output_dir}", file=sys.stderr)


def cmd_sts(args: argparse.Namespace) -> int:
    deps = _lazy_imports()
    InputExample = deps["InputExample"]
    losses = deps["losses"]

    hf_path = "alina0195/romteb-ro-sts"
    revision = _read_revision(hf_path)
    train = _load_split(hf_path, args.split, revision)

    examples = [_rosts_examples(row, InputExample=InputExample) for row in train]
    print(f"[finetune] RoSTS train examples: {len(examples)}", file=sys.stderr)

    _train(
        model_id=args.model,
        output_dir=Path(args.output),
        examples=examples,
        loss_ctor=losses.CosineSimilarityLoss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
    )
    return 0


def cmd_pair(args: argparse.Namespace) -> int:
    deps = _lazy_imports()
    InputExample = deps["InputExample"]
    losses = deps["losses"]

    hf_path = "alina0195/romteb-ronli"
    revision = _read_revision(hf_path)
    train = _load_split(hf_path, args.split, revision)

    # Match RoNLIPairClassification.dataset_transform: keep only Entailment (1)
    # and Contrastive (0); drop Consequence (2) and Unrelated (3).
    kept = [row for row in train if int(row["label"]) in (0, 1)]
    examples = [_ronli_examples(row, InputExample=InputExample) for row in kept]
    print(
        f"[finetune] RoNLI train examples (Ent/Con only): {len(examples)}",
        file=sys.stderr,
    )

    def _loss(model):
        return losses.ContrastiveLoss(model=model, margin=args.margin)

    _train(
        model_id=args.model,
        output_dir=Path(args.output),
        examples=examples,
        loss_ctor=_loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finetune_encoder",
        description=(
            "Fine-tune a sentence encoder on RoSTS or RoNLI to produce the "
            "'not-frozen' variant for the RoMTEB dual-encoder comparison."
        ),
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="HF id or local encoder dir")
    common.add_argument("--output", required=True, help="Checkpoint directory")
    common.add_argument("--split", default="train")
    common.add_argument("--epochs", type=int, default=3)
    common.add_argument("--batch_size", type=int, default=16)
    common.add_argument("--warmup_ratio", type=float, default=0.1)

    subparsers = parser.add_subparsers(dest="task", required=True)
    p_sts = subparsers.add_parser(
        "sts",
        parents=[common],
        help="Fine-tune on RoSTS (cosine regression)",
    )
    p_sts.set_defaults(func=cmd_sts)

    p_pair = subparsers.add_parser(
        "pair",
        parents=[common],
        help="Fine-tune on RoNLI Ent/Contrastive (contrastive loss)",
    )
    p_pair.add_argument("--margin", type=float, default=0.5)
    p_pair.set_defaults(func=cmd_pair)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
