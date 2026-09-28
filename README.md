# RoMTEB

Romanian Massive Text Embedding Benchmark. It follows the same idea as
[MTEB](https://github.com/embeddings-benchmark/mteb): a sentence encoder
is scored on classification, pair classification, STS, retrieval,
reranking and bitext mining, all in Romanian.

Where MMTEB already has a Romanian subset, we reuse that task. The rest
are custom datasets, pinned to a commit SHA so a re-run hits the same
rows. Clustering is in the catalog but skipped by the default protocol.

The encoder is **frozen** by default. Classification is the one place
something gets trained on top of the frozen vectors: a class-weighted
logistic regression probe on the full train split (the old 8-shot ×
10-draw protocol is kept as a `k8` sidecar). For STS and pair
classification we also publish a **fine-tuned** row (`_ft` variant)
alongside the frozen one so the frozen→fine-tuned delta is visible on
the leaderboard; retrieval, reranking and bitext mining stay
frozen-only. Details in
[`docs/encoder_finetuning.md`](docs/encoder_finetuning.md).

## Tasks

| Type | Task | Source | License |
|---|---|---|---|
| Classification | `RoABSAClassification` | `upb-nlp/RoABSA` (aspect-level) | CC-BY-4.0 |
| Classification | `RoOffenseClassification` | `readerbench/news-ro-offense` | CC-BY-4.0 |
| Classification | `HateSpeechROClassification` | `readerbench/ro-hate-speech` | CC-BY-NC-4.0 |
| Classification | `REDv2EmotionClassification` | REDv2 | CC-BY-4.0 |
| Classification | `SciTechBanROClassification` | ClickbaitSciTechRO | not specified |
| Classification | `SaRoCoClassification` | SaRoCo | not specified |
| Classification | `HistNERoMentionClassification` | `avramandrei/histnero` | MIT |
| Classification | `MassiveIntentClassification` (ron) | MTEB | inherits |
| Classification | `MassiveScenarioClassification` (ron) | MTEB | inherits |
| Classification | `SIB200Classification` (ron_Latn) | MTEB | inherits |
| Classification | `RomanianSentimentClassification.v2` | MTEB | inherits |
| Classification | `RoNLIClassification` | `Eduard6421/RONLI` (4 classes) | CC-BY-NC-SA-4.0 |
| PairClassification | `RoNLIPairClassification` | `Eduard6421/RONLI` | CC-BY-NC-SA-4.0 |
| STS | `RoSTS` | `dumitrescustefan/ro_sts` | CC-BY-SA-4.0 |
| Retrieval | `XQuADRetrieval` (ro) | MTEB / `google/xquad` | inherits |
| Retrieval | `RoDTALLawsRetrieval` | `GRAI-UNSTPB/RoD-TAL` | CC-BY-NC-SA-4.0 |
| Retrieval | `WebFAQRetrieval` (ron) | MTEB | inherits |
| Retrieval | `WikipediaRetrievalMultilingual` (ro) | MTEB | inherits |
| Retrieval | `BelebeleRetrieval` (ron_Latn) | MTEB | inherits |
| Retrieval | `MQARoCQARetrieval` | CLiPS MQA CQA (`ro-cqa-question`) | CC0 |
| Retrieval (MCQ) | `JuRoLegalExamRetrieval` | JuRo | not specified |
| Retrieval (MCQ) | `WWTBMRoQARetrieval` | `WWTBM/wwtbm` | not specified |
| Retrieval (MCQ) | `RoMedQAv2Retrieval` | `craciuncg/RoMedQA_v2` | Apache-2.0 |
| Retrieval (MCQ) | `GrileGrammarRetrieval` | GRILE | not specified |
| Reranking | `WikipediaRerankingMultilingual` (ro) | MTEB | inherits |
| BitextMining | `NTREXBitextMining` (ron_Latn) | MTEB | inherits |
| BitextMining | `Tatoeba` (ron) | MTEB | inherits |
| BitextMining | `IWSLT2017BitextMining` (en-ro) | MTEB | inherits |

`RoABSAClassification` is aspect-level (`Entitate: {aspect}` plus the
review), not document polarity. `HistNERoMentionClassification` types a
gold span; it is not IOB NER.

RoNLI shows up twice on purpose. `RoNLIPairClassification` is the MTEB
XNLI-style binary probe (Entailment vs Contrastive, cosine AP).
`RoNLIClassification` keeps all four upstream classes (Contrastive,
Entailment, Consequence, Unrelated); premise and hypothesis are
embedded separately and a logistic-regression probe is trained on the
SentEval pair features `[u, v, |u-v|, u*v]` (set
`ROMTEB_RONLI_PROBE=3part` for the `[u, v, |u-v|]` ablation). Both
`RoNLIClassification` and `RoABSAClassification` report macro-F1
because the test splits are heavily imbalanced. They share a source
dataset, so domain / dataset means never double-count RoNLI.

Legal / medical / grammar MCQs (JuRo, RoMedQA, WWTBM, GRILE) are
scored as retrieval with a per-question candidate pool (`top_ranked`
holds only that question's options). Structurally this is the
reranking protocol; they are grouped under Retrieval per the Sep 2026
taxonomy consolidation so the leaderboard has one MCQ column instead
of two. The full-option-bank variants were dropped in the same audit
(three collapsed to p90 nDCG@10 < 0.10 because the option-bank corpus
has duplicate surface forms). `MSMarcoRoRetrieval` is held out for
training and is not a leaderboard task. Bitext mining is a cross-
lingual section: it gets its own table and does not enter the Overall
score.

`romteb-eval --list-tasks` prints the live roster as JSON (no GPU).

## Install

Python 3.10+, a GPU for dense models, and a Hugging Face token if any
of the datasets are gated.

```bash
pip install -e .
```

Run this from the repo root. `romteb-eval` is then on PATH. `python -m romteb` does the same thing.
Custom dataset revisions live in `tasks/_revisions.json`; override one
with `ROMTEB_REV__<owner>__<name>=<sha>` if you need to.

## Run

Needs CUDA. On CPU a dense encode looks hung; `--allow_cpu` exists for
debugging, not for a real eval. BM25 (`--model romteb/bm25s-ro`) is
lexical and does not need a GPU.

Smoke (RoSTS only):

```bash
romteb-eval --model intfloat/multilingual-e5-large --preset smoke --output results/_smoke
```

Official v1 protocol (`full` is the default; clustering is skipped):

```bash
romteb-eval --model intfloat/multilingual-e5-large --loader auto --output results
```

`--preset core` is classification + pair classification + STS, no IR.
`--tasks RoSTS RoABSAClassification …` replaces the preset if you only
want a subset.

### Classification: full-train vs k-shot

The v1 protocol trains a class-weighted logistic regression on the
entire train split. Every run also emits an 8-shot sidecar
(`<Task>.k8.json`, MTEB's default probe, 10 draws) so numbers can be
compared with legacy MTEB reports.

```bash
# Default: full-train probe + 8-shot sidecar side by side.
romteb-eval --model intfloat/multilingual-e5-large --preset core --output results

# Full-train only (skip the 8-shot sidecar).
romteb-eval --model intfloat/multilingual-e5-large --preset core \
  --skip_k8 --output results/full_train_only

# Legacy 8-shot × 10 draws only, on one task.
romteb-eval --model intfloat/multilingual-e5-large \
  --tasks RoABSAClassification --skip_k8 --output results/k8_only
# then override the sample budget for a k-shot sweep:
ROMTEB_K_SHOT=8 romteb-eval --tasks RoABSAClassification \
  --model intfloat/multilingual-e5-large --skip_k8 --output results/k8_sweep

# RoNLI probe ablation: swap the 4-part SentEval features for the
# 3-part Sentence-BERT variant.
ROMTEB_RONLI_PROBE=3part romteb-eval \
  --model intfloat/multilingual-e5-large \
  --tasks RoNLIClassification --output results/ronli_3part
```

`ROMTEB_K_SHOT` (see `romteb/eval_config.py::classification_shots`)
lets you pin every classification task to a fixed shot count for a
learning-curve sweep; without it the default is full-train.

### Encoder handling: frozen vs fine-tuned

| Family | Frozen (default) | Fine-tuned (`_ft` variant) |
|---|---|---|
| Classification | LR probe on frozen vectors | not run — protocol scores the representation, not a downstream classifier |
| PairClassification (RoNLI) | cosine AP, no training | `scripts/finetune_encoder.py pair` → contrastive loss on RoNLI Ent/Con |
| STS (RoSTS) | cosine Spearman, no training | `scripts/finetune_encoder.py sts` → cosine-similarity regression |
| Retrieval / Reranking / BitextMining | frozen bi-encoder | not run |

Fine-tune then evaluate the checkpoint the usual way:

```bash
# STS fine-tune, then eval; the _ft suffix keeps the row separate from the frozen one.
python scripts/finetune_encoder.py sts \
    --model intfloat/multilingual-e5-large \
    --output ckpt/e5-large_rosts_ft --epochs 3 --batch_size 16
romteb-eval --model ckpt/e5-large_rosts_ft --tasks RoSTS \
    --output results/e5-large_rosts_ft

# RoNLI pair fine-tune (Entailment vs Contrastive, ContrastiveLoss).
python scripts/finetune_encoder.py pair \
    --model intfloat/multilingual-e5-large \
    --output ckpt/e5-large_ronli_ft --epochs 3 --margin 0.5
romteb-eval --model ckpt/e5-large_ronli_ft \
    --tasks RoNLIPairClassification --output results/e5-large_ronli_ft
```

Frozen and fine-tuned rows are never mixed inside the same aggregate.
Full policy and the "why frozen for classification" argument are in
[`docs/encoder_finetuning.md`](docs/encoder_finetuning.md).

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required (unless `--list-tasks` / `--summary-only`) | Hub id, local encoder directory, or `romteb/bm25s-ro` |
| `--output` | `results` | MTEB JSON tree plus `summary.json` |
| `--preset` | `full` | `smoke` / `core` / `full` |
| `--tasks` | unset | Restrict to these task names (overrides `--preset`) |
| `--loader` | `auto` | `st` = SentenceTransformer + declared prefixes; `mteb` = official wrappers (Qwen3, jina, …) |
| `--no_flash_attn` | off | Needed for Granite / ModernBERT on some images |
| `--no_trust_remote_code` | off | Refuse custom modeling code (recommended for untrusted uploads) |
| `--batch_size` | 32 | 1–4 if a large model OOMs |
| `--skip_k8` | off | Skip the 8-shot classification sidecar |
| `--allow_cpu` | off | Allow dense models on CPU (debug only) |
| `--overwrite_results` | off | Re-run tasks that already have a JSON |
| `--summary` | `<output>/summary.json` | Single-model payload for a caller |
| `--summary-only` | off | Rebuild that JSON from results already on disk |
| `--list-tasks` | off | Print the roster as JSON and exit (no GPU) |
| `--job-id` | unset | Copied into `summary.json` |

E5-style models need `query:` / `passage:` prefixes in
`romteb/model_prompts.py`. A missing prefix does not crash. Retrieval
just comes out several points low.

Granite:

```bash
romteb-eval --model ibm-granite/granite-embedding-97m-multilingual-r2 \
  --loader st --no_flash_attn --output results
```

Docker (same CLI, GPU):

```bash
docker build -t romteb-eval:0.1.0 .
docker run --gpus all --rm \
  -v "$MODEL_DIR:/model:ro" -v "$OUT:/out" -v "$HF_CACHE:/hf" \
  -e HF_HOME=/hf -e HF_TOKEN \
  romteb-eval:0.1.0 \
    --model /model --preset full --output /out
```

Mount a Hub cache if you have one. Without it, every job re-downloads
the datasets.

## Output

```
<output>/
  summary.json                         one model: status, per-task scores, by_type
  <org>__<model>/<rev>/<Task>.json     MTEB result
  <org>__<model>/<rev>/<Task>.k8.json  classification at 8 shots, when protocol k ≠ 8
  predictions/<org>__<model>/          IR rankings + classification / pair predictions
```

`scripts/bootstrap_classification.py` resamples the per-example
classification / pair predictions to report accuracy (and macro-F1
where applicable) with 95% CIs — the full-train replacement for the
old 10-draw sample std. `scripts/ronli_baselines.py` prints chance,
Jaccard word-overlap, and a negation-cue baseline on the RoNLI
Entailment / Contrastive test split.

`summary.json` is for a single-model run (a platform job, a PR check).
`type_macro` averages Classification, PairClassification, STS,
Retrieval and Reranking after dropping tasks flagged by the audit
(`SATURATED_TASKS` / `OVERALL_EXCLUDE_TASKS` in
`romteb/eval_config.py`). Bitext mining is reported separately.
`overall_borda` is `null` in a single-model summary: Borda is a rank
against other models, so it only exists after you aggregate a roster.

```bash
romteb-eval --summary-only --model intfloat/multilingual-e5-large --output results
romteb-aggregate --results_dir results
```

After a run, `scripts/aggregate_results.py` writes the leaderboard
tables next to `--output` (Overall is category Borda, lower is better).
Bitext is reported separately and does not enter Overall.

## Auditing tasks that do not discriminate

A task belongs on the leaderboard only if it ranks encoders
differently. `scripts/audit_task_discrimination.py` reads the
`results/` tree and flags four failure modes:

1. **BM25 saturation** — BM25 ≥ 0.85 primary score AND the best dense
   encoder is within 0.05 of BM25 (Retrieval / Reranking only). The
   task is lexically solvable; the encoder is not being tested.
2. **Random-baseline collapse** — every model ≤ the empirical random
   baseline in `RERANKING_RANDOM_BASELINE` (MCQ only). Nobody solves
   it; the mean would just add noise.
3. **Ceiling collapse** — top-vs-bottom span on the primary metric
   ≤ 0.02. Every model scores the same; the task cannot rank them.
4. **Above-floor collapse** — Classification only: every model ≥ 0.90
   AND the span ≤ 0.03. The task is easy and does not discriminate
   (the LaRoSeDa failure mode).

```bash
python scripts/audit_task_discrimination.py \
    --results results \
    --bm25-slug romteb__bm25s-ro \
    --json audit_report.json
```

The script prints a markdown table with the failure mode and the
raw numbers for every flagged task. Nothing is auto-mutated —
`SATURATED_TASKS` and `OVERALL_EXCLUDE_TASKS` in
`romteb/eval_config.py` are still edited by hand after a human
sanity-checks the report. Removed tasks (like LaRoSeDa) come out of
`REUSED_TASK_NAMES` in `romteb/benchmark.py` entirely.

Criteria, thresholds, and current findings live in
[`docs/task_audit.md`](docs/task_audit.md).

## Adding a task

Prep script under `romteb/data_prep/` that pushes a Hub dataset and
writes the SHA into `romteb/tasks/_revisions.json`; a task class under
`romteb/tasks/<type>/`; wire it into `_CUSTOM_CLASSES` (or
`REUSED_TASK_NAMES`) in `romteb/benchmark.py`, add its reporting
domain / dataset key in `romteb/eval_config.py`, then smoke it on
`intfloat/multilingual-e5-large` and re-run the discrimination audit
(`scripts/audit_task_discrimination.py`) before adding it to the
public roster.

## Further reading

- [`docs/encoder_finetuning.md`](docs/encoder_finetuning.md) — frozen
  vs fine-tuned policy per task family, with the fine-tune → eval
  workflow.
- [`docs/reused_mteb_tasks.md`](docs/reused_mteb_tasks.md) — provenance
  audit of every reused MTEB task (annotation source, label count,
  translation status, risks).
- [`docs/task_audit.md`](docs/task_audit.md) — the discrimination
  criteria (BM25 saturation, random-baseline collapse, span / floor
  collapse), the audit script, and the current findings.

## License

Code is Apache-2.0. Datasets keep their own licenses (see the table).
Some sources are CC-BY-NC; check before you train on the eval sets.

Built on [MTEB](https://github.com/embeddings-benchmark/mteb).
