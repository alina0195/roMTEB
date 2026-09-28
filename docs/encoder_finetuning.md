# Encoder handling: frozen vs fine-tuned

RoMTEB has three families of tasks where the question "is the encoder
frozen?" has a substantive answer. This note fixes the policy per
family and points at the code and CLI paths for each mode.

## Classification (probe only)

- **Encoder**: **frozen**. Always.
- **Trained head**: a class-weighted logistic-regression probe on the
  extracted vectors (`romteb.eval_config.make_probe`,
  `class_weight="balanced"`).
- **Rationale**: the entire benchmark is designed to score *the
  representation*, not a downstream classifier. Fine-tuning the encoder
  on labelled train data would let a small model win by memorisation
  rather than by having a stronger embedding space.
- **How to run**: default. `romteb-eval --preset core ...` fits the
  probe once (full-train) and emits an 8-shot sidecar for comparison
  with older MTEB numbers.

The pair-features variant (`RoNLIClassification`) still keeps the
encoder frozen — it swaps the input from one text to a SentEval-style
`[u, v, |u-v|, u*v]` pair vector before the probe.

## PairClassification (RoNLI: both modes)

- **Frozen (default RoMTEB path)**: cosine AP on frozen embeddings, no
  training. Reported as `RoNLIPairClassification` with `main_score =
  max_ap`.

- **Fine-tuned (`_ft` variant)**: metric-learning objective on RoNLI
  train (Entailment vs Contrastive), then eval on the same test split.

  ```bash
  # 1. Fine-tune (produces a sentence-transformers checkpoint).
  python scripts/finetune_encoder.py pair \
      --model intfloat/multilingual-e5-large \
      --output ckpt/e5-large_ronli_ft \
      --epochs 3 --batch_size 16 --margin 0.5

  # 2. Evaluate the fine-tuned checkpoint the usual way.
  romteb-eval \
      --model ckpt/e5-large_ronli_ft \
      --tasks RoNLIPairClassification \
      --output results/e5-large_ronli_ft
  ```

  The eval CLI is unchanged; it just loads the checkpoint from disk.
  Aggregator scripts key on the model directory name, so the `_ft`
  suffix keeps the frozen and fine-tuned rows separate.

- **Reporting**: both rows are shown in the leaderboard, side by side.
  The frozen row is the model's representation quality; the delta
  frozen→fine-tuned is the RoNLI-specific ceiling that fine-tuning
  reaches with modest supervision. Do not fold the two into a single
  aggregate.

## STS (RoSTS: both modes)

- **Frozen (default)**: Spearman correlation between cosine similarity
  of frozen embeddings and gold [0, 5] scores. Reported as `RoSTS`
  with `main_score = cosine_spearman`.

- **Fine-tuned (`_ft` variant)**: cosine-similarity regression on the
  RoSTS train split.

  ```bash
  python scripts/finetune_encoder.py sts \
      --model intfloat/multilingual-e5-large \
      --output ckpt/e5-large_rosts_ft \
      --epochs 3 --batch_size 16

  romteb-eval \
      --model ckpt/e5-large_rosts_ft \
      --tasks RoSTS \
      --output results/e5-large_rosts_ft
  ```

  Same reporting rule: keep the frozen and fine-tuned rows separate.

## What is *not* fine-tuned

- The four MCQ Retrieval tasks (JuRo / WWTBM / RoMedQA / GRILE).
- Open-corpus retrieval (RoDTAL, WebFAQ, Wikipedia, XQuAD, Belebele,
  MQA-CQA).
- Bitext mining.

These stay frozen-only. If we ever add a fine-tuned retriever variant
(e.g. contrastive training on WebFAQ-like data), it goes through the
same `_ft`-suffix convention so nothing is silently averaged.

## Convention summary

| Family | Frozen | Fine-tuned |
|---|---|---|
| Classification | default (LR probe) | not run |
| PairClassification (RoNLI) | default (cosine AP) | `_ft` variant via `finetune_encoder.py pair` |
| STS (RoSTS) | default (cosine Spearman) | `_ft` variant via `finetune_encoder.py sts` |
| Retrieval / Reranking / BitextMining | default | not run |

Frozen and fine-tuned rows are never mixed inside the same aggregate.
