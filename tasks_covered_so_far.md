
## Task table (current roster)

See also [`README.md`](README.md), [`docs/task_audit.md`](docs/task_audit.md),
and [`docs/reused_mteb_tasks.md`](docs/reused_mteb_tasks.md).

| Category | Task name | Source | Status |
|---|---|---|---|
| Classification | `RoABSAClassification` | RoABSA aspect-level | custom |
| Classification | `RoOffenseClassification` | news-ro-offense | custom |
| Classification | `HateSpeechROClassification` | hate-speech-ro | custom |
| Classification | `REDv2EmotionClassification` | REDv2 | custom |
| Classification | `SciTechBanROClassification` | ClickbaitSciTechRO xlsx | custom |
| Classification | `SaRoCoClassification` | SaRoCo | custom |
| Classification | `HistNERoMentionClassification` | HistNERo span→type | custom |
| Classification | `RoNLIClassification` | RONLI (4 classes, SentEval [u,v,\|u−v\|,u·v] probe) | custom |
| Classification | `MassiveIntentClassification` (ron) | MTEB | reuse |
| Classification | `MassiveScenarioClassification` (ron) | MTEB | reuse |
| Classification | `SIB200Classification` (ron_Latn) | MTEB | reuse |
| Classification | `RomanianSentimentClassification.v2` | MTEB | reuse |
| PairClassification | `RoNLIPairClassification` | RONLI (cosine AP, no LR) | custom |
| STS | `RoSTS` | RO-STS | custom |
| Retrieval | `XQuADRetrieval` (ro) | MTEB | reuse |
| Retrieval | `RoDTALLawsRetrieval` | RoD-TAL | custom |
| Retrieval | `WebFAQRetrieval` (ron) | MTEB | reuse |
| Retrieval | `WikipediaRetrievalMultilingual` (ro) | MTEB | reuse |
| Retrieval | `BelebeleRetrieval` (ron_Latn) | MTEB | reuse |
| Retrieval | `MQARoCQARetrieval` | clips/mqa CQA (ro) | custom |
| Retrieval (MCQ) | `JuRoLegalExamRetrieval` | JuRo per-question option pool | custom |
| Retrieval (MCQ) | `WWTBMRoQARetrieval` | WWTBM per-question option pool | custom |
| Retrieval (MCQ) | `RoMedQAv2Retrieval` | RoMedQA per-question option pool | custom |
| Retrieval (MCQ) | `GrileGrammarRetrieval` | GRILE per-question option pool | custom |
| Reranking | `WikipediaRerankingMultilingual` (ro) | MTEB | reuse |
| BitextMining† | `NTREXBitextMining` / `Tatoeba` / `IWSLT2017BitextMining` | MTEB | reuse |
| Clustering | `SIB200ClusteringS2S` | MTEB | present in catalog, skipped by default preset |

† BitextMining is a cross-lingual section, reported separately and
excluded from the Overall score.

## Recently removed / consolidated

| Task | Date | Reason |
|---|---|---|
| `MSMarcoRoRetrieval` | — | Held out for training; not a benchmark task. |
