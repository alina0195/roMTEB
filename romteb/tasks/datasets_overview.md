# RoMTEB benchmark datasets

Datasets currently registered in `romteb/benchmark.py`.

Note: `FloresBitextMining`, `RoDTALLawsPairClassification`, `XQuADRoRetrieval`
and `RonSTS` were removed/consolidated (see README). Retired Tier-2
classification tasks are also excluded. BitextMining is a cross-lingual section,
excluded from the Overall score.

| Task | Type | Source | Language | Test instances | Example |
|---|---|---|---|---:|---|
| `IWSLT2017BitextMining` | BitextMining | https://aclanthology.org/2017.iwslt-1.1/ (`mteb/IWSLT2017BitextMining`) | Multilingual (bitext pairs incl. Romanian) | 6 | ERROR: 0 |
| `NTREXBitextMining` | BitextMining | https://huggingface.co/datasets/davidstap/NTREX (`mteb/NTREXBitextMining`) | Multilingual (bitext pairs incl. Romanian) | 1997 | s1="?" s2="?" |
| `Tatoeba` | BitextMining | https://github.com/facebookresearch/LASER/tree/main/data/tatoeba/v1 (`mteb/tatoeba-bitext-mining`) | Multilingual (bitext pairs incl. Romanian) | 1 | ERROR: 0 |
| `HateSpeechROClassification` | Classification | https://huggingface.co/datasets/readerbench/ro-hate-speech (`alina0195/romteb-hatespeech-ro`) | Romanian | ? | ERROR: Dataset  doesn't exist on the Hub or cannot be accessed. |
| `MassiveIntentClassification` | Classification | https://arxiv.org/abs/2204.08582 (`mteb/amazon_massive_intent`) | Romanian (derived/translated from other languages) | 1 | ERROR: 0 |
| `MassiveScenarioClassification` | Classification | https://arxiv.org/abs/2204.08582 (`mteb/amazon_massive_scenario`) | Romanian (derived/translated from other languages) | 1 | ERROR: 0 |
| `Moroco.v2` | Classification | https://huggingface.co/datasets/moroco (`mteb/moroco`) | Romanian | 2048 | text=""$NE$ incurajat evaziunea fiscala", spun acum inspectorii $NE$ care fac verificari . Zeci de camioane cu flori au intra…" label=1 |
| `REDv2EmotionClassification` | Classification | https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets/tree/main/REDv2 (`alina0195/romteb-red-v2-emotion`) | Romanian | 757 | text="Ce panică!" label=2 |
| `RoABSAClassification` | Classification | https://huggingface.co/datasets/upb-nlp/RoABSA (`alina0195/romteb-roabsa`) | Romanian | ? | ERROR: Dataset  doesn't exist on the Hub or cannot be accessed. |
| `RoOffenseClassification` | Classification | https://huggingface.co/datasets/readerbench/news-ro-offense (`alina0195/romteb-ro-offense`) | Romanian | ? | ERROR: Dataset doesn't exist on the Hub or cannot be accessed. |
| `RomanianReviewsSentiment.v2` | Classification | https://arxiv.org/abs/2101.04197 (`mteb/romanian_reviews_sentiment`) | Romanian | 2048 | text="materiale de calitate, telecomanda detasabila. are foarte multe plusuri. eu am asteptat 2 saptamani, sa ajunga din sofi…" label=3 |
| `RomanianSentimentClassification.v2` | Classification | https://arxiv.org/abs/2009.08712 (`mteb/romanian_sentiment`) | Romanian | 2048 | text="unul dintre cele mai bune filme vreodata, ideea unei interpretari duble implica noi toti. <br /> <br /> ar fi prot un s…" label=1 |
| `SIB200Classification` | Classification | https://arxiv.org/abs/2309.07445 (`mteb/sib200`) | Romanian (derived/translated from other languages) | 1 | ERROR: 0 |
| `SIB200ClusteringS2S` | Clustering | https://arxiv.org/abs/2309.07445 (`mteb/sib200`) | Romanian (derived/translated from other languages) | 1 | ... |
| `JuRoLegalExamPairClassification` | PairClassification | https://github.com/craciuncg/GRAF/tree/main/JuRo (`alina0195/romteb-juro-legal-pair-classification`) | Romanian | 5188 | s1="[Drept Civil] 2. În materia filiaţiei:" s2="acţiunea în stabilirea paternităţii din afara căsătoriei nu poate fi pornită împotriva moştenitorilor pretinsului tată" score=0 |
| `RoMedQAv2PairClassification` | PairClassification | https://huggingface.co/datasets/craciuncg/RoMedQA_v2 (`alina0195/romteb-romedqa-v2-pair-classification`) | Romanian | 8495 | s1="Care dintre următorii hormoni stimulează evacuarea bilei din vezica biliară:" s2="colecistochinina" score=1 |
| `RoNLIPairClassification` | PairClassification | https://github.com/Eduard6421/RONLI (`alina0195/romteb-ronli`) | Romanian (derived/translated from other languages) | 1122 | s1="Avantajul unui comandant în tanc s-a dovedit a fi esențial în tactica germană de folosire a diviziilor blindate Panzer." s2="Panzer iii era în realitate pe câmpul de luptă mult superior unor tancuri mai bine dotate, care aveau însă 2 tanchiști …" score=2 |
| `WWTBMRoQAPairClassification` | PairClassification | https://arxiv.org/abs/2506.05991 (`alina0195/romteb-wwtbm-ro-pair-classification`) | Romanian | 4000 | s1="Ce ingredient este indispensabil la prepararea piftiei?" s2="praful de copt" score=0 |
| `WebFAQRetrieval` | Retrieval | https://huggingface.co/PaDaS-Lab (`mteb/WebFAQRetrieval`) | Romanian (derived/translated from other languages) | 0 | ['ron'] |
| `WikipediaRetrievalMultilingual` | Retrieval | https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries (`mteb/WikipediaRetrievalMultilingual`) | Romanian (derived/translated from other languages) | 0 | ['ro'] |
| `XQuADRetrieval` | Retrieval | https://huggingface.co/datasets/xquad (`google/xquad`) | Romanian (derived/translated from other languages) | 1184 | query="Câte puncte a cedat apărarea echipei Panthers?" |
| `RoSTS` | STS | https://huggingface.co/datasets/dumitrescustefan/ro_sts (`alina0195/romteb-ro-sts`) | Romanian (derived/translated from other languages) | 1379 | s1="O fată își aranjează părul." s2="O fată se piaptănă." score=2.5 |

