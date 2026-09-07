# MTEB Romanian task inventory

Generated from `mteb==2.12.30`. Edit the `decision` and `reason`
columns by hand after this script runs; rerun to refresh the list.

| task_name | type | ron_subsets | hf_path | main_score | decision | reason |
|---|---|---|---|---|---|---|
| `XM3600T2IRetrieval` | Any2AnyMultilingualRetrieval | ro | `mteb/xm3600` | ndcg_at_10 | reuse | default; review and override if needed |
| `CommonVoiceMini17A2TRetrieval` | Any2AnyRetrieval | frold, ro | `mteb/common_voice_17_0_mini` | hit_rate_at_5 | reuse | default; review and override if needed |
| `CommonVoiceMini17T2ARetrieval` | Any2AnyRetrieval | frold, ro | `mteb/common_voice_17_0_mini` | hit_rate_at_5 | reuse | default; review and override if needed |
| `CommonVoiceMini21A2TRetrieval` | Any2AnyRetrieval | rm-sursilv, rm-vallader, ro | `mteb/common_voice_21_0_mini` | hit_rate_at_5 | reuse | default; review and override if needed |
| `CommonVoiceMini21T2ARetrieval` | Any2AnyRetrieval | rm-sursilv, rm-vallader, ro | `mteb/common_voice_21_0_mini` | hit_rate_at_5 | reuse | default; review and override if needed |
| `FleursA2TRetrieval` | Any2AnyRetrieval | ro_ro | `mteb/fleurs` | hit_rate_at_5 | reuse | default; review and override if needed |
| `FleursT2ARetrieval` | Any2AnyRetrieval | ro_ro | `mteb/fleurs` | hit_rate_at_5 | reuse | default; review and override if needed |
| `SIBFLEURS` | AudioClassification | ron_Latn | `mteb/sib-fleurs-multilingual-mini` | accuracy | reuse | default; review and override if needed |
| `BibleNLPBitextMining` | BitextMining | eng_Latn-ron_Latn, ron_Latn-eng_Latn, eng_Latn-roo_Latn, roo_Latn-eng_Latn, eng_Latn-rop_Latn, rop_Latn-eng_Latn, eng_Latn-row_Latn, row_Latn-eng_Latn, eng_Latn-rro_Latn, rro_Latn-eng_Latn, eng_Latn-wro_Latn, wro_Latn-eng_Latn | `mteb/biblenlp-corpus` | f1 | reuse | default; review and override if needed |
| `FloresBitextMining` | BitextMining | ... | `mteb/FloresBitextMining` | f1 | exclude | only a devtest split in mteb 2.12.x; no usable score |
| `IWSLT2017BitextMining` | BitextMining | en-ro, it-ro, nl-ro, ro-en, ro-it, ro-nl | `mteb/IWSLT2017BitextMining` | f1 | reuse | default; review and override if needed |
| `NTREXBitextMining` | BitextMining | cat_Latn-ron_Latn, eng_Latn-ron_Latn, fra_Latn-ron_Latn, glg_Latn-ron_Latn, ita_Latn-ron_Latn, mlt_Latn-ron_Latn, por_Latn-ron_Latn, ron_Latn-cat_Latn, ron_Latn-eng_Latn, ron_Latn-fra_Latn, ron_Latn-glg_Latn, ron_Latn-ita_Latn, ron_Latn-mlt_Latn, ron_Latn-por_Latn, ron_Latn-spa_Latn, spa_Latn-ron_Latn | `mteb/NTREXBitextMining` | f1 | reuse | default; review and override if needed |
| `Tatoeba` | BitextMining | ron-eng | `mteb/tatoeba-bitext-mining` | f1 | reuse | default; review and override if needed |
| `WebFAQBitextMiningQAs` | BitextMining | dan-ron, deu-ron, fra-ron, ita-ron, nld-ron, nor-ron, por-ron, ron-spa, ron-swe, eng-ron | `PaDaS-Lab/webfaq-bitexts` | f1 | reuse | default; review and override if needed |
| `WebFAQBitextMiningQuestions` | BitextMining | dan-ron, deu-ron, fra-ron, ita-ron, nld-ron, nor-ron, por-ron, ron-spa, ron-swe, eng-ron | `PaDaS-Lab/webfaq-bitexts` | f1 | reuse | default; review and override if needed |
| `MassiveIntentClassification` | Classification | ro | `mteb/amazon_massive_intent` | accuracy | reuse | default; review and override if needed |
| `MassiveScenarioClassification` | Classification | ro | `mteb/amazon_massive_scenario` | accuracy | reuse | default; review and override if needed |
| `Moroco.v2` | Classification | default | `mteb/moroco` | accuracy | exclude | obsolete; dropped from RoMTEB |
| `RomanianReviewsSentiment.v2` | Classification | default | `mteb/romanian_reviews_sentiment` | accuracy | reuse | default; review and override if needed |
| `RomanianSentimentClassification.v2` | Classification | default | `mteb/romanian_sentiment` | accuracy | reuse | default; review and override if needed |
| `SIB200Classification` | Classification | ron_Latn | `mteb/sib200` | accuracy | reuse | default; review and override if needed |
| `SIB200ClusteringS2S` | Clustering | ron_Latn | `mteb/sib200` | v_measure | reuse | default; review and override if needed |
| `MultiEURLEXMultilabelClassification` | MultilabelClassification | ro | `mteb/eurlex-multilingual` | accuracy | reuse | default; review and override if needed |
| `WikipediaRerankingMultilingual` | Reranking | ro | `mteb/WikipediaRerankingMultilingual` | map_at_1000 | reuse | default; review and override if needed |
| `BelebeleRetrieval` | Retrieval | ron_Latn-ron_Latn, ron_Latn-eng_Latn, eng_Latn-ron_Latn | `mteb/belebele` | ndcg_at_10 | reuse | default; review and override if needed |
| `WebFAQRetrieval` | Retrieval | ron | `mteb/WebFAQRetrieval` | ndcg_at_10 | reuse | default; review and override if needed |
| `WikipediaRetrievalMultilingual` | Retrieval | ro | `mteb/WikipediaRetrievalMultilingual` | ndcg_at_10 | reuse | default; review and override if needed |
| `XQuADRetrieval` | Retrieval | ro | `google/xquad` | ndcg_at_10 | reuse | default; review and override if needed |
| `RonSTS` | STS | default | `mteb/RonSTS` | cosine_spearman | custom | same RO-STS data as custom RoSTS; RonSTS dropped from RoMTEB |
