# RoMTEB

Romanian Massive Text Embedding Benchmark. It follows the same idea as
[MTEB](https://github.com/embeddings-benchmark/mteb): a frozen sentence
encoder is scored on classification, STS, retrieval, reranking, and a
few related tasks, all in Romanian.

Where MMTEB already has a Romanian subset, we reuse that task. The rest
are custom datasets, pinned to a commit SHA so a re-run hits the same
rows. Clustering is in the catalog but skipped by the default protocol.
Summarization is not in v1.

The encoder itself is never fine-tuned. Classification is the one place
something gets trained: a logistic regression probe on the extracted
vectors (few shots per class, 10 draws).

## Tasks

| Type | Task | Source | License |
|---|---|---|---|
| Classification | `RoABSAClassification` | `upb-nlp/RoABSA` (aspect-level) | CC-BY-4.0 |
| Classification | `RoOffenseClassification` | `readerbench/news-ro-offense` | CC-BY-4.0 |
| Classification | `HateSpeechROClassification` | `readerbench/ro-hate-speech` | CC-BY-NC-4.0 |
| Classification | `REDv2EmotionClassification` | REDv2 | CC-BY-4.0 |
| Classification | `RoMathDomainClassification` | `cosmadrian/romath` | CC-BY-NC-4.0 |
| Classification | `SciTechBanROClassification` | ClickbaitSciTechRO | not specified |
| Classification | `SaRoCoClassification` | SaRoCo | not specified |
| Classification | `HistNERoMentionClassification` | `avramandrei/histnero` | MIT |
| Classification | `MassiveIntentClassification` (ron) | MTEB | inherits |
| Classification | `MassiveScenarioClassification` (ron) | MTEB | inherits |
| Classification | `SIB200Classification` (ron_Latn) | MTEB | inherits |
| Classification | `RomanianReviewsSentiment.v2` | MTEB | inherits |
| Classification | `RomanianSentimentClassification.v2` | MTEB | inherits |
| PairClassification | `RoNLIPairClassification` | `Eduard6421/RONLI` | CC-BY-4.0 |
| STS | `RoSTS` | `dumitrescustefan/ro_sts` | CC-BY-SA-4.0 |
| Retrieval | `XQuADRetrieval` (ro) | MTEB / `google/xquad` | inherits |
| Retrieval | `RoDTALLawsRetrieval` | `GRAI-UNSTPB/RoD-TAL` | CC-BY-NC-SA-4.0 |
| Retrieval | `WebFAQRetrieval` (ron) | MTEB | inherits |
| Retrieval | `WikipediaRetrievalMultilingual` (ro) | MTEB | inherits |
| Retrieval | `MQARoCQARetrieval` | CLiPS MQA CQA (`ro-cqa-question`) | CC0 |
| Reranking | `JuRoLegalExamReranking` | JuRo | not specified |
| Reranking | `WWTBMRoQAReranking` | `WWTBM/wwtbm` | not specified |
| Reranking | `RoMedQAv2Reranking` | `craciuncg/RoMedQA_v2` | Apache-2.0 |
| Reranking | `GrileGrammarReranking` | GRILE | not specified |
| BitextMining | `NTREXBitextMining` (ron_Latn) | MTEB | inherits |
| BitextMining | `Tatoeba` (ron) | MTEB | inherits |
| BitextMining | `IWSLT2017BitextMining` (en-ro) | MTEB | inherits |

`RoABSAClassification` is aspect-level (`Entitate: {aspect}` plus the
review), not document polarity. `HistNERoMentionClassification` types a
gold span; it is not IOB NER.

Legal / medical / grammar MCQs (JuRo, RoMedQA, WWTBM, GRILE) are
reranking over the option pool. `MSMarcoRoRetrieval` is held out for
training and is not a leaderboard task. Bitext mining is a
cross-lingual section: it gets its own table and does not enter the
Overall score.

`python -m romteb --list-tasks` prints the live roster as JSON.

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

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | Hub id, local encoder directory, or `romteb/bm25s-ro` |
| `--output` | `results` | MTEB JSON tree plus `summary.json` |
| `--preset` | `full` | `smoke` / `core` / `full` |
| `--loader` | `auto` | `st` = SentenceTransformer + declared prefixes; `mteb` = official wrappers (Qwen3, jina, …) |
| `--no_flash_attn` | off | needed for Granite / ModernBERT on some images |
| `--no_trust_remote_code` | off | turn this on for weights you did not train |
| `--batch_size` | 32 | 1–4 if a large model OOMs |
| `--overwrite_results` | off | re-run tasks that already have a JSON |
| `--summary` | `<output>/summary.json` | single-model payload for a caller |
| `--summary-only` | off | rebuild that JSON from results already on disk |
| `--job-id` | unset | copied into `summary.json` |

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
  predictions/<org>__<model>/          IR rankings (bootstrap CI)
```

`summary.json` is for a single-model run (a platform job, a PR check).
`type_macro` averages Classification, PairClassification, STS,
Retrieval, and Reranking after dropping saturated / overall-excluded
tasks. `overall_borda` is `null` there: Borda is a rank against other
models, so it only exists after you aggregate a roster.

```bash
python -m romteb --summary-only --model intfloat/multilingual-e5-large --output results
romteb-aggregate --results_dir results
```

After a run, `scripts/aggregate_results.py` writes the leaderboard
tables next to `--output` (Overall is category Borda, lower is better).
Bitext is reported separately and does not enter Overall.

## Adding a task

Prep script that pushes a Hub dataset and writes the SHA, a task class
under `romteb/tasks/<type>/`, wire it in `eval_config.py` and
`benchmark.py`, then smoke it on `multilingual-e5-large`.

## License

Code is Apache-2.0. Datasets keep their own licenses (see the table).
Some sources are CC-BY-NC; check before you train on the eval sets.

Built on [MTEB](https://github.com/embeddings-benchmark/mteb).
