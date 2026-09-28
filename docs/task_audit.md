# Task discrimination audit

A task belongs on the leaderboard only if it **ranks encoders
differently**. This note fixes the criteria we use to demote or
remove a task, the script that runs the audit automatically, and the
current findings.

## Failure modes

A task is a candidate for demotion (per-task table only, dropped
from Overall / type-macro / domain means) or full removal when any
of these hold:

1. **BM25 saturation** — Retrieval / Reranking only.
   BM25 alone already hits an absolute primary score ≥ **0.85** AND
   the best dense encoder is within **0.05** of BM25. Lexical
   overlap solves it, so the task is not testing the embedding.

2. **Random-baseline collapse** — MCQ only.
   Every model's primary score is ≤ the empirical random baseline
   (`RERANKING_RANDOM_BASELINE` in `romteb/eval_config.py`, computed
   over uniform random permutations that account for multi-gold
   rate). Nobody solves it; the mean would just add noise.

3. **Ceiling collapse (Δ span)** — any task type.
   Top-vs-bottom model gap on the primary metric is ≤ **0.02**. All
   encoders score essentially the same; the task cannot rank them.
   Common cause: dataset size too small for meaningful cell-level
   spread, or the labels are so coarse that any encoder above
   random hits the same plateau.

4. **Above-floor collapse** — Classification only.
   Every model scores ≥ **0.90** on the primary metric AND the Δ
   span is ≤ **0.03**. The task is easy and does not discriminate.
   This is the LaRoSeDa failure mode: star→sentiment labels are so
   correlated with surface tokens that any modern encoder plateaus
   near the ceiling.

Thresholds are the audit's defaults; every flag prints its numbers
so a human can adjust for a specific family (e.g. tighter span for
STS where 0.02 Spearman is meaningful).

## How to run

```bash
python scripts/audit_task_discrimination.py \
    --results results \
    --bm25-slug romteb__bm25s-ro \
    --json audit_report.json
```

The script reads `results/<slug>/<rev>/<Task>.json` for every model
in the roster, computes the primary metric, and prints a markdown
table of flagged tasks with the failure mode and the raw numbers.
Nothing is auto-mutated — `SATURATED_TASKS` and
`OVERALL_EXCLUDE_TASKS` in `romteb/eval_config.py` are still edited
by hand after a human sanity-checks the report.

Tweak thresholds per family:

```bash
# stricter Δ-span for STS (Spearman moves in small increments):
python scripts/audit_task_discrimination.py --span-eps 0.01

# looser BM25 gate — flag anything BM25 wins:
python scripts/audit_task_discrimination.py --bm25-abs 0.75 --bm25-eps 0.03
```

## What gets done with a flag

- **Saturation flag** → add task to `SATURATED_TASKS`. Kept in
  per-task tables so the audit finding is visible; excluded from
  Overall / type-macro / domain means.
- **Random-collapse flag** → add task to `OVERALL_EXCLUDE_TASKS`.
  Same visibility rule (per-task only).
- **Ceiling / above-floor collapse** → remove the task outright
  from the benchmark (edit `REUSED_TASK_NAMES` in
  `romteb/benchmark.py` or delete the custom task file). Also
  update `TASK_DATASET`, `TASK_REPORTING_DOMAIN`, and
  `CLASSIFICATION_SHOTS` so the aggregate never looks it up.

The removal path is stricter than the demotion path: demoted tasks
still cost compute at eval time; removed tasks free that budget for
a task that actually discriminates.

## Current findings (Sep 2026)

| Task | Mode | Action | Notes |
|---|---|---|---|
| `XQuADRetrieval` | BM25 saturation | Demoted (`SATURATED_TASKS`) | BM25 ≥ 0.85 nDCG@10; top dense encoder < 0.05 above. Short lexically overlapping passages. |
| `JuRoLegalExamRetrieval` | Random-baseline collapse | Demoted (`OVERALL_EXCLUDE_TASKS`) | Multi-gold random baseline is 0.639 map@1000; every encoder sits on top of it. |
| MCQ full-option-bank Retrieval variants (JuRo / WWTBM / RoMedQA / GRILE, deprecated) | Ceiling collapse + duplicate surface forms | **Removed** | Three collapsed to p90 nDCG@10 < 0.10; the WWTBM one was driven by a single outlier. See the older commit for the exact numbers. |
| `RomanianReviewsSentiment.v2` (LaRoSeDa) | Above-floor collapse + heuristic labels | **Removed** (Sep 2026) | Every modern encoder ≥ 0.90 accuracy; span across the roster is inside the sample-std noise band. Labels are star→sentiment (no human adjudication), so a small gap could be label noise, not real signal. |

The audit re-runs on every leaderboard refresh. If a new task hits
any threshold, this table gets updated and the config follows.
