# RoMTEB benchmark datasets

Datasets currently registered in `romteb/benchmark.py`. Regenerate this
table by running the roster dump:

    romteb-eval --list-tasks

Note: `Moroco.v2`, `RomanianReviewsSentiment.v2` (LaRoSeDa),
`RoMathDomainClassification`, `FloresBitextMining`,
`RoDTALLawsPairClassification`, `XQuADRoRetrieval`, `RonSTS`, the four
MCQ `*Reranking` classes, the four MCQ full-option-bank `*Retrieval`
v1 variants, and the JuRo / RoMedQA / WWTBM PairClassification tasks
were all removed or consolidated. See
[`docs/task_audit.md`](../../docs/task_audit.md) and
[`docs/reused_mteb_tasks.md`](../../docs/reused_mteb_tasks.md).

BitextMining is a cross-lingual section, excluded from the Overall
score.

| Task | Type | Source | Language |
|---|---|---|---|
| `RoABSAClassification` | Classification | https://huggingface.co/datasets/upb-nlp/RoABSA (`alina0195/romteb-roabsa`) | Romanian |
| `RoOffenseClassification` | Classification | https://huggingface.co/datasets/readerbench/news-ro-offense (`alina0195/romteb-ro-offense`) | Romanian |
| `HateSpeechROClassification` | Classification | https://huggingface.co/datasets/readerbench/ro-hate-speech (`alina0195/romteb-hatespeech-ro`) | Romanian |
| `REDv2EmotionClassification` | Classification | https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets/tree/main/REDv2 (`alina0195/romteb-red-v2-emotion`) | Romanian |
| `SciTechBanROClassification` | Classification | ClickbaitSciTechRO (`alina0195/romteb-scitechbanro`) | Romanian |
| `SaRoCoClassification` | Classification | SaRoCo (`alina0195/romteb-saroco`) | Romanian |
| `HistNERoMentionClassification` | Classification | `avramandrei/histnero` (`alina0195/romteb-histnero-mentions`) | Romanian |
| `RoNLIClassification` | Classification | https://github.com/Eduard6421/RONLI (`alina0195/romteb-ronli`, all 4 classes) | Romanian |
| `MassiveIntentClassification` | Classification | https://arxiv.org/abs/2204.08582 (`mteb/amazon_massive_intent`) | Romanian (human-translated) |
| `MassiveScenarioClassification` | Classification | https://arxiv.org/abs/2204.08582 (`mteb/amazon_massive_scenario`) | Romanian (human-translated) |
| `SIB200Classification` | Classification | https://arxiv.org/abs/2309.07445 (`mteb/sib200`) | Romanian (human-translated) |
| `RomanianSentimentClassification.v2` | Classification | https://arxiv.org/abs/2009.08712 (`mteb/romanian_sentiment`) | Romanian |
| `RoNLIPairClassification` | PairClassification | https://github.com/Eduard6421/RONLI (`alina0195/romteb-ronli`, Ent vs Con) | Romanian |
| `RoSTS` | STS | https://huggingface.co/datasets/dumitrescustefan/ro_sts (`alina0195/romteb-ro-sts`) | Romanian (human-translated) |
| `XQuADRetrieval` | Retrieval | https://huggingface.co/datasets/xquad (`google/xquad`) | Romanian (human-translated) |
| `RoDTALLawsRetrieval` | Retrieval | RoD-TAL (`alina0195/romteb-rod-tal-retrieval`) | Romanian |
| `WebFAQRetrieval` | Retrieval | https://huggingface.co/PaDaS-Lab (`mteb/WebFAQRetrieval`) | Romanian |
| `WikipediaRetrievalMultilingual` | Retrieval | https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries (`mteb/WikipediaRetrievalMultilingual`) | Romanian (LLM-generated queries) |
| `BelebeleRetrieval` | Retrieval | https://arxiv.org/abs/2308.16884 (`mteb/belebele`) | Romanian (human-translated) |
| `MQARoCQARetrieval` | Retrieval | https://huggingface.co/datasets/clips/mqa (`alina0195/romteb-mqa-ro-cqa-retrieval`) | Romanian |
| `JuRoLegalExamRetrieval` | Retrieval (MCQ) | https://github.com/craciuncg/GRAF/tree/main/JuRo (`alina0195/romteb-juro-legal-reranking`) | Romanian |
| `WWTBMRoQARetrieval` | Retrieval (MCQ) | https://arxiv.org/abs/2506.05991 (`alina0195/romteb-wwtbm-ro`) | Romanian |
| `RoMedQAv2Retrieval` | Retrieval (MCQ) | https://huggingface.co/datasets/craciuncg/RoMedQA_v2 (`alina0195/romteb-romedqa-v2-reranking`) | Romanian |
| `GrileGrammarRetrieval` | Retrieval (MCQ) | GRILE (`alina0195/romteb-grile-reranking`) | Romanian |
| `WikipediaRerankingMultilingual` | Reranking | `mteb/WikipediaRerankingMultilingual` | Romanian (LLM-generated queries) |
| `NTREXBitextMining` | BitextMining | https://huggingface.co/datasets/davidstap/NTREX (`mteb/NTREXBitextMining`) | Multilingual (bitext pairs incl. Romanian) |
| `Tatoeba` | BitextMining | https://github.com/facebookresearch/LASER/tree/main/data/tatoeba/v1 (`mteb/tatoeba-bitext-mining`) | Multilingual (bitext pairs incl. Romanian) |
| `IWSLT2017BitextMining` | BitextMining | https://aclanthology.org/2017.iwslt-1.1/ (`mteb/IWSLT2017BitextMining`) | Multilingual (bitext pairs incl. Romanian) |
