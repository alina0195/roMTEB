# RoMTEB — protocol decisions and task revision notes

_Status: recommendations, 26 Sep 2026. Based on roMTEB@3ad6c93 (7 Sep 2026), mteb main (25 Sep 2026), mteb 1.0.0 tag. Update as decisions land._

## 1. Classification: full-train probe vs k-shot

**What MTEB actually does.** Since v1.0.0 the probe is k-shot: `samples_per_label=8` × `n_experiments=10`, `LogisticRegression(max_iter=100)`, no C tuning (some tasks override to 16/32/128). The MTEB paper text only says "train set embeddings are used to train a logistic regression" — it never mentions k-shot, so the protocol is under-documented upstream.

**What others did.**
- k-shot (MTEB default family): PL-MTEB (8×10; rationale: results "less influenced by the training data and more by the encoding method"), SEB (16/label, with replacement, 10 draws), C-MTEB (32/label), ruMTEB (8×10), MTEB-NL (8–128/label).
- Full train: **JMTEB v1** (full train split; LR and kNN at default hyper-params, best picked on validation macro-F1, reported on test, no caps); **SentEval** (full train, L2 tuned on validation / nested CV).

**Does `-1` work?** No. `_undersample_data` keeps a row only while `label_counter[label] < samples_per_label`; with `-1` nothing is selected → empty train → sklearn error. Use a sentinel larger than any class count (e.g. `10**9`).

**Recommended protocol.**
- Primary: full-train linear probe. `n_experiments=1` (every draw is identical with full data; the 10 runs would be wasted), `LogisticRegression(max_iter=1000)` (100 iterations won't converge on 10k+ × 1024-d), fixed C (document it; if tuned, tune on validation only).
- Secondary: keep the existing `.k8.json` sidecar for MTEB-comparable numbers. Wrap the model in `mteb.models.cache_wrappers.CachedEmbeddingWrapper` so the sidecar reuses the full-train embeddings (k8 rows are a subset) — near-zero extra cost.
- Paper figure: learning curve k ∈ {8, 32, 128, full} from one encode → sample-efficiency analysis per model.
- Variance: replace the 10-draw std with a test-set bootstrap CI (reuse `scripts/bootstrap_ir.py` logic).
- Imbalance: k-shot trains on balanced samples; full train learns the label prior. Report macro-F1 beside accuracy for imbalanced sets (HateSpeech, RoOffense, RoABSA, REDv2), or make macro-F1 the main score for them.
- Caveats to state in paper: scores compress near the ceiling; high-dimensional embeddings get a larger probe (d × C params); not comparable to MTEB-leaderboard numbers for reused tasks (MASSIVE, SIB-200, reviews) → that's what the k8 sidecar is for.
- Optional stratified cap (e.g. 20k) if a train split is huge; JMTEB used none.

## 2. Pair classification

**RoNLI label bug (found 26 Sep 2026).**
- Upstream RONLI ids, per `corpus/NLI_extractor.py` @ fd75ce0: 0 = Contrastive, 1 = Entailment, 2 = Consequence, 3 = Unrelated (random pairs); confirmed by `word_to_class` in `dataset/generate_datasets/check_outputs.py`.
- The old `prep_ronli.py` assumed 0 = entailment, 1 = neutral, 2 = contradiction and silently dropped id 3. The task therefore scored contrast (as "positive") against reasoning, with the reasoning side undersampled. The "92.8% contradiction" that motivated rebalancing was really reasoning.
- Old GitHub fallback also globbed `train_easy/hard/curriculum*`, `old_validation.json` and a second `validation.json` into the splits.
- Fix:
  - Keep the upstream ids.
  - Test = validation ∪ test, deduplicated: 103 duplicate pairs merged, 29 label conflicts dropped.
  - Binary entailment vs contrast gives 162 / 141 = 303 test pairs, with no rebalancing.
  - Train pushed verbatim.
  - License corrected to CC BY-NC-SA 4.0.
- RoNLI is now used by two tasks from the same Hub dataset:
  - `RoNLIPairClassification`: Entailment vs Contrastive, cosine max_ap, no probe.
  - `RoNLIClassification`: all 4 classes, SentEval-style probe on [u, v, |u−v|, u·v], full train, macro-F1.
  - `TASK_DATASET` maps both to "RoNLI" so they are not double-counted in domain means.
- Phase-3 training: use train id 1 (entailment) as positive and id 0 (contrast) as the hard negative. The validation file is now part of test, so never tune on it.


- Only RoNLI remains, with 148 test pairs (74/74 after rebalancing) → AP is very noisy.
  - Don't undersample; AP handles imbalance (MTEB XNLI does not rebalance). Report prevalence.
  - Pair classification trains nothing, so val + test can both be eval (if val is not used in phase-3 training).
  - Re-check the RONLI label map: ~6k manually annotated val/test pairs shrinking to 148 suggests most labels are being dropped.
- Candidate additions:
  - Bible Paraphrase Corpus: same verse in two versions = positive, adjacent verse = hard negative (gold by construction; archaic register).
  - TaPaCo-ro paraphrases + BM25-mined high-overlap non-paraphrases (verify a sample).
  - Duplicate-question PC from MQA/WebFAQ FAQ questions across sites (SprintDuplicateQuestions template), LLM-verified.
  - **Romanian-specific minimal pairs (novel):** diacritic/cedilla variants of the same sentence = positive; diacritic-sensitive minimal pairs (fata/fată/fața, peste/pește, sa/să/șa), negation, inflection/word-order swaps (PAWS-style) = negative. LLM-generated + human-verified. Directly tests the tokenizer thesis (RQ1).
  - Romanian Paraphrase Dataset (~100k, undocumented): only after auditing a sample.
- Several small sets could be one task with subsets (MTEB multi-subset pattern) to stabilise the category mean.

## 3. Retrieval

- BM25 gap (existing results): XQuAD saturated (excluded), Wikipedia near-saturated (BM25 0.847 vs best 0.915); MQA-CQA, RoD-TAL and WebFAQ still separate models.
- Quick wins already in mteb with a `ron` subset: **BelebeleRetrieval** (ron_Latn-ron_Latn) and **WikipediaRerankingMultilingual** (ro) — **registered** (Sep 2026). Still open: MultiEURLEX multilabel (ro); RoITD (marked [done] in the inventory xlsx but not in `benchmark.py`).
- RoITD / any SQuAD-style set will saturate like XQuAD unless the corpus is enlarged with distractors (all ro Wikipedia IT paragraphs) or it is used as reranking with hard negatives.
- New tasks from existing sources:
  - RO Text Summarization: summary → article (72k corpus, gold by construction).
  - News: title → body.
  - Recipes: name/ingredients → preparation.
  - Legal: RoD-TAL / JuRo question → MARCELL article (bigger distractor corpus).
  - **Long-document retrieval** over RO-Stories / ELTeC chapters (LLM questions): tests the 2k/4k/8k context question in RQ3.
  - ROCODE: Romanian problem statement → solution (text-to-code).
- Synthetic query rules:
  - Use more than one generator LLM, and avoid one family only (Qwen3-Embedding was trained on Qwen-synthesised data).
  - Prompt for low lexical overlap.
  - Filter with an LLM judge for answerability, and human-validate a sample (report agreement).
  - Keep a private test slice (RTEB-style).
- Dedup MQA-FAQ against WebFAQ (both from Common Crawl FAQPage markup) to avoid double counting.

## 4. Reranking

> **Update (Sep 2026):** the four MCQ classes (JuRo, WWTBM, RoMedQA,
> GRILE) were renamed `*Retrieval` and moved to
> `romteb/tasks/retrieval/` per the taxonomy consolidation
> (per-question `top_ranked` pool is retained; only the type label
> changed). The Reranking column in the leaderboard now holds
> `WikipediaRerankingMultilingual` (reused MTEB) only. The build
> options below still apply and are how the column grows.

- **Difference from retrieval.**
  - Retrieval ranks the whole corpus (first-stage recall).
  - Reranking ranks a short candidate list per query (10–100 items, all topically related, typically the top-k of a first-stage retriever), so it measures fine-grained discrimination among hard negatives.
  - Metrics: MAP@1000 (MTEB default), MRR@10, nDCG@10.
- **Format.**
  - In mteb v2, reranking = retrieval + `top_ranked` {qid: [candidate doc ids]}, in BEIR layout: corpus / queries / qrels / top_ranked. The repo already uses this.
  - Legacy format: {query, positive[], negative[]}.
- **How others built reranking sets.**
  - SciDocsRR: citations. AskUbuntu / StackOverflowDupQuestions: duplicate questions. MindSmall: clicks.
  - ESCI: graded E/S/C/I product labels.
  - T2Reranking: human-graded Sogou passages.
  - MMarco: translated.
  - MTEB-French Syntec/Alloprof: retrieval data + BM25 top-10 non-relevant as negatives. The authors admit this correlates reranking with retrieval.
  - JQaRA: 100 candidates pooled by RRF over 5 embedders, positives LLM- and human-verified.
  - PL-MTEB had none (no Polish data).
- **Current RoMTEB reranking** is MCQ option pools (3–5 options). This is really "exam/knowledge option ranking": distractors are designed to be plausible, and embeddings encode similarity rather than truth. That is why JuRo sits at the random baseline. Keep it as a separate labelled sub-category, not canonical reranking.
- **Build options, by value/cost:**
  1. Reuse WikipediaRerankingMultilingual (ro). No cost.
  2. Convert RoD-TAL, MQA-CQA, WebFAQ-ro, RoITD into reranking:
     - Pool the top-30 from BM25-ro plus 2–3 dense models from different families; never your own model.
     - LLM-judge the pooled candidates to catch false negatives, and human-check a sample.
     - Count the reranking task and its source retrieval task as one dataset in aggregation (`TASK_DATASET` already supports this).
  3. Structured hard negatives:
     - WebFAQ: same-site answers.
     - Summarization: same-category, same-week articles. Watch out for multi-outlet coverage of the same event (false negatives).
     - Recipes: same category.
     - Bible: adjacent verses.
  4. Evidence reranking for MCQ: question (+ answer) → candidate law articles (RoD-TAL, JuRo). Real reranking, legal domain.
  5. ESCI-style e-commerce reranking from internal search logs. High novelty and practical value, but only with internal approval:
     - Product/query text only.
     - Scrub PII in queries.
     - k-anonymity threshold on queries.
     - Route through legal@emag.ro / security@emag.ro.

## 5. Clustering (assumed "not just those 2")

- RoNewsOutlet/RoNewsType P2P are imported in `tasks/clustering/__init__.py`, but their modules are missing from the repo.
- Candidates:
  - ~~RoMath domain (7 labels, S2S).~~ RoMath was removed from the
    RoMTEB scope entirely (Sep 2026); do not add it back.
  - MOROCO topics (6 labels, P2P). Note the `$NE$` masking.
  - Romanian Categorized Web Dataset.
  - RO Text Summarization / News categories.
  - SciTechBanRO.
  - ROST authorship (10 authors: style clustering, a distinct signal).
- Use mteb v2 fast clustering (bootstrapped subsets, v-measure).

## 6. Delivery

- There is no `MTEB(ron, v1)` in mteb yet (existing language benchmarks include pol, fra, deu, rus, jpn, nld, kor, por, slk, spa, fas, tha, cmn), so RoMTEB would be the first.
  - Upstream the custom tasks into `mteb/tasks/*/ron/` and register `MTEB(ron, v1)`.
  - Results then flow through embeddings-benchmark/results to the public leaderboard.
  - Keep the `romteb` package as a thin layer for the extras: Borda, domain slices, bootstrap CIs, BM25-ro, full-train protocol.
- Move datasets to an org namespace, write full dataset cards, and keep revisions pinned.
- Licensing blockers for redistribution: JuRo, WWTBM, GRILE, SciTechBanRO and SaRoCo are "not specified". Get written permission from the authors or ship prep scripts only.
- Private held-out slices of new synthetic sets → submit to RTEB (there is RTEB(fra/deu/jpn) but no Romanian).
- Paper: benchmark + protocol analysis (full vs k-shot learning curves, BM25-gap saturation audit, MCQ-as-reranking negative result).