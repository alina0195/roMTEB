
## Task table (current roster)

See also [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

| Category | Task name | Source | Status |
|---|---|---|---|
| Classification | `RoABSAClassification` | RoABSA aspect-level | custom |
| Classification | `RoOffenseClassification` | news-ro-offense | custom |
| Classification | `HateSpeechROClassification` | hate-speech-ro | custom |
| Classification | `RoMathDomainClassification` | RoMath | custom |
| Classification | `REDv2EmotionClassification` | REDv2 | custom |
| Classification | `SciTechBanROClassification` | ClickbaitSciTechRO xlsx | custom |
| Classification | `SaRoCoClassification` | SaRoCo | custom |
| Classification | `HistNERoMentionClassification` | HistNERo span→type | custom |
| Classification | `MassiveIntentClassification` (ron) | MTEB | reuse |
| Classification | `MassiveScenarioClassification` (ron) | MTEB | reuse |
| Classification | `SIB200Classification` (ron_Latn) | MTEB | reuse |
| Classification | `RomanianReviewsSentiment.v2` | MTEB | reuse |
| Classification | `RomanianSentimentClassification.v2` | MTEB | reuse |
| PairClassification | `RoNLIPairClassification` | RONLI (cosine AP, no LR) | custom |
| STS | `RoSTS` | RO-STS | custom |
| Retrieval | `XQuADRetrieval` (ro) | MTEB | reuse |
| Retrieval | `RoDTALLawsRetrieval` | RoD-TAL | custom |
| Retrieval | `WebFAQRetrieval` (ron) | MTEB | reuse |
| Retrieval | `WikipediaRetrievalMultilingual` (ro) | MTEB | reuse |
| Retrieval | `MSMarcoRoRetrieval` | alina0195/ro-msmarco | custom |
| Retrieval | `MQARoCQARetrieval` | clips/mqa CQA (ro) | custom |
| Retrieval | `JuRoLegalExamRetrieval` | JuRo option bank | custom |
| Retrieval | `WWTBMRoQARetrieval` | WWTBM option bank | custom |
| Retrieval | `RoMedQAv2Retrieval` | RoMedQA option bank | custom |
| Retrieval | `GrileGrammarRetrieval` | GRILE option bank | custom |
| Reranking | `JuRoLegalExamReranking` | JuRo option pool | custom |
| Reranking | `WWTBMRoQAReranking` | WWTBM option pool | custom |
| Reranking | `RoMedQAv2Reranking` | RoMedQA option pool | custom |
| Reranking | `GrileGrammarReranking` | GRILE option pool | custom |
| BitextMining† | `NTREXBitextMining` / `Tatoeba` / `IWSLT2017BitextMining` | MTEB | reuse |
| Clustering | `SIB200ClusteringS2S` | MTEB | reuse |

Dropped: `Moroco.v2`; JuRo/WWTBM/RoMedQA PairClassification (archived scripts).
