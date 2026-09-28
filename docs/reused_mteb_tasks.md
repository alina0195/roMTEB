# Reused MTEB tasks — provenance audit

We inherit 13 Romanian task subsets from MTEB. This note records, for
each one, where the data comes from, how the labels or targets were
produced, and whether the Romanian side is native, translated, or LLM-
generated. The intent is to make it obvious which cells are trustworthy
signal vs which are noisier and should be interpreted with care.

Legend for the **Source** column
- **Native Romanian** – written by Romanian speakers, no translation step
- **Human-translated** – translated from another language, with human review
- **MT / LLM-generated** – machine translation or LLM output, minimal review
- **Cross-lingual bitext** – professionally translated parallel corpus

## Classification (4 reused)

| Task | Labels | Source | Annotation | Notes / risks |
|---|---:|---|---|---|
| `MassiveIntentClassification` (ron) | 60 | Human-translated from English seed (Amazon MASSIVE, Bastianelli et al., 2022) | Utterances collected in English, professionally translated to Romanian; intent labels are language-agnostic | Test 2974 / dev 2033 / train 11514. Some Romanian utterances are stiff MT-ese; intent granularity is fine (60 classes for a spoken assistant). Used mostly to check that the encoder separates fine-grained intents. |
| `MassiveScenarioClassification` (ron) | 18 | Same MASSIVE data | Human labels on the English source; carried over to translations | Shares text and split with the intent task, so `TASK_DATASET` marks them both as `MASSIVE` to avoid double-counting a single source in domain/type means. |
| `SIB200Classification` (ron_Latn) | 7 | Human-translated (FLORES-200 devtest → SIB-200 topical labels; Adelani et al., 2023) | Sentence-level topics annotated on the English side; Romanian is one of 200 FLORES translations | Test 204 / dev 99 / train 701. Small test set — noisy per-class means. Topics: science/tech, travel, politics, sports, health, entertainment, geography. |

**Summary.** All four have human-provided labels. MASSIVE and SIB-200
are translated. There is no LLM-generated label anywhere in the classification section.

## Retrieval (4 reused)

| Task | Corpus size / queries | Source | Annotation | Notes / risks |
|---|---|---|---|---|
| `XQuADRetrieval` (ro) | 1190 queries / 240 passages | Human-translated (SQuAD 1.1 → XQuAD, Artetxe et al., 2020) | Passage + question + gold answer span translated by professional translators | Flagged as saturated in our audit (`SATURATED_TASKS`): BM25 already scores ≥ 0.85 nDCG@10 because the passages are short and lexically overlapping. Kept in per-task tables, excluded from Overall / type-macro. |
| `WebFAQRetrieval` (ron) | ~ hundreds of Q/A pairs | **Native Romanian** — WebFAQ, PaDaS-Lab crawl of FAQ pages | Q/A pairs harvested from real FAQ pages, no human relabelling | Realistic FAQ domain, but crawl-derived: some pairs are boilerplate or duplicates. |
| `WikipediaRetrievalMultilingual` (ro) | ~ paragraphs from ro.wikipedia | **MT / LLM-generated queries** — ellamind/wikipedia-2023-11-retrieval-multilingual-queries | LLM-generated queries over Wikipedia paragraphs (paragraphs are native) | Query side is model output, so it can favour retrievers that share the query model's phrasing bias. Use with care. |
| `BelebeleRetrieval` (ron_Latn) | 900 queries | Human-translated (Belebele, Bandarkar et al., 2024) | 900 multi-choice reading-comprehension items translated from a shared English source; retrieval variant recasts each item as query→passage | Passages are FLORES-200 translations (professional). Retrieval framing is a MTEB adaptation, not part of the original release. |

## Reranking (1 reused)

| Task | Queries / candidates | Source | Annotation | Notes / risks |
|---|---|---|---|---|
| `WikipediaRerankingMultilingual` (ro) | same wiki source as retrieval variant | LLM-generated queries over Wikipedia paragraphs | Query side is LLM output; positives = the source paragraph, negatives = nearest-neighbour paragraphs | `TASK_DATASET` groups this with `WikipediaRetrievalMultilingual` under the same `Wikipedia` key so it is never double-counted in a domain mean. |

## Bitext Mining (3 reused, cross-lingual)

| Task | Pairs | Source | Notes |
|---|---:|---|---|
| `NTREXBitextMining` (ron_Latn) | 1997 | News-Test 2019 (Barrault et al., 2019), professional translations | Solid signal; used broadly across multilingual benchmarks. |
| `Tatoeba` (ron) | ~1000 sentence pairs | Community-sourced (`facebookresearch/LASER/tree/main/data/tatoeba/v1`) | Community translations of everyday sentences; quality is uneven at the pair level but the aggregate is fine. |
| `IWSLT2017BitextMining` (en-ro) | 6 (test) | TED talk transcripts, professional translations (Cettolo et al., 2017) | Tiny test partition — MTEB uses 6 pairs. Kept for lineage; do not read too much into a single-model ranking here. |

Bitext mining is a **cross-lingual** section: it is reported in its
own table and excluded from the Overall score.

## Overall stance

- Every reused label / target has a human somewhere in the loop, either
  in the source-language annotation (MASSIVE, SIB-200, XQuAD, Belebele)
  or in the corpus construction (LaRoSeDa, WebFAQ, NTREX).
- **Translated** tasks: MASSIVE, SIB-200, XQuAD, Belebele, NTREX,
  IWSLT2017. These are trustworthy as evaluations, but the Romanian
  side inherits the source distribution — do not read them as evidence
  about *native* Romanian phrasing.
- **LLM-generated query** tasks: `WikipediaRetrievalMultilingual`,
  `WikipediaRerankingMultilingual`. Signal is real but biased toward
  retrievers that share the query LLM's phrasing.
- **Saturated**: `XQuADRetrieval` is dominated by lexical overlap; kept
  for transparency but excluded from Overall.
- **Tiny test**: `IWSLT2017BitextMining` (6 pairs) and Tatoeba (~1k)
  should be read as ordinal, not calibrated.

If a task above ever loses its human-annotation provenance (upstream
switches to LLM labels, for example), drop it from the leaderboard or
demote it to descriptive-only. The current roster passes.
