"""Fast dataset stats for RoMTEB markdown table (targeted loads only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from romteb._hf_datasets import Dataset, load_dataset

from romteb.data_prep._classification_utils import stratified_train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "romteb" / "docs" / "dataset_stats.json"


def cls_from_source(path: str, text_keys=("text", "sentence", "comment", "Tweet")):
    raw = load_dataset(path)
    split = "train" if "train" in raw else list(raw.keys())[0]
    rows = []
    for r in raw[split]:
        text = next((r.get(k) for k in text_keys if r.get(k)), None)
        label = r.get("label")
        if text and label is not None:
            rows.append({"text": str(text).strip(), "label": label})
    test = stratified_train_test_split(Dataset.from_list(rows))["test"]
    ex = test[0]
    return len(test), f'text="{ex["text"][:90]}..." label={ex["label"]}'


def main():
    stats = {}

    stats["RoABSAClassification"] = cls_from_source("upb-nlp/RoABSA")
    stats["RoOffenseClassification"] = cls_from_source("readerbench/news-ro-offense")
    stats["HateSpeechROClassification"] = cls_from_source("readerbench/ro-hate-speech")

    sts = load_dataset("dumitrescustefan/ro_sts", split="test")
    r0 = sts[0]
    stats["RoSTS"] = (
        len(sts),
        f's1="{r0.get("sentence1", r0.get("sentence_a"))[:80]}..." score={r0.get("score", r0.get("label"))}',
    )

    # XQuAD-ro is reused from MMTEB as XQuADRetrieval (the custom
    # XQuADRoRetrieval duplicate was removed).
    xq = load_dataset("google/xquad", "xquad.ro", split="validation")
    stats["XQuADRetrieval"] = (
        len(xq),
        f'query="{xq[0]["question"][:80]}..." gold_doc="{xq[0]["context"][:80]}..."',
    )

    sib = load_dataset("mteb/sib200", "ron_Latn", split="test")
    stats["SIB200Classification"] = (
        len(sib),
        f'text="{sib[0]["text"][:80]}..." label={sib[0]["label"]}',
    )
    stats["SIB200ClusteringS2S"] = (len(sib), stats["SIB200Classification"][1])

    massive = load_dataset("mteb/amazon_massive_intent", "ro", split="test")
    stats["MassiveIntentClassification"] = (
        len(massive),
        f'text="{massive[0]["text"][:80]}..." label={massive[0]["label"]}',
    )
    massive_s = load_dataset("mteb/amazon_massive_scenario", "ro", split="test")
    stats["MassiveScenarioClassification"] = (
        len(massive_s),
        f'text="{massive_s[0]["text"][:80]}..." label={massive_s[0]["label"]}',
    )

    # FloresBitextMining removed: it only exposes a 'devtest' split and never
    # produced a usable score, so it is not part of RoMTEB.

    tatoeba = load_dataset("mteb/tatoeba-bitext-mining", "ron-eng", split="test")
    stats["Tatoeba"] = (
        len(tatoeba),
        f's1="{tatoeba[0]["sentence1"][:80]}..." s2="{tatoeba[0]["sentence2"][:80]}..."',
    )

    iwslt = load_dataset("mteb/IWSLT2017BitextMining", "ro-en", split="test")
    stats["IWSLT2017BitextMining"] = (
        len(iwslt),
        f's1="{iwslt[0]["sentence1"][:80]}..." s2="{iwslt[0]["sentence2"][:80]}..."',
    )

    ntrex = load_dataset("mteb/NTREXBitextMining", "ron_Latn-eng_Latn", split="test")
    stats["NTREXBitextMining"] = (
        len(ntrex),
        f's1="{ntrex[0]["sentence1"][:80]}..." s2="{ntrex[0]["sentence2"][:80]}..."',
    )

    webfaq_q = load_dataset("mteb/WebFAQRetrieval", "ron", "queries", split="test")
    webfaq_c = load_dataset("mteb/WebFAQRetrieval", "ron", "corpus", split="test")
    stats["WebFAQRetrieval"] = (
        len(webfaq_q),
        f'query="{webfaq_q[0]["text"][:80]}..." gold_doc="{webfaq_c[0]["text"][:80]}..."',
    )

    wiki_q = load_dataset("mteb/WikipediaRetrievalMultilingual", "ro", "queries", split="test")
    wiki_c = load_dataset("mteb/WikipediaRetrievalMultilingual", "ro", "corpus", split="test")
    stats["WikipediaRetrievalMultilingual"] = (
        len(wiki_q),
        f'query="{wiki_q[0]["text"][:80]}..." gold_doc="{wiki_c[0]["text"][:80]}..."',
    )

    path = ROOT / "datasets" / "MedQARo Dataset" / "romedqa_test_dataset.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("Intrebare") or "").strip()
            c = (row.get("Epicriza") or "").strip()
            if q and c:
                rows.append((q, c))
    stats["MedQARoRetrieval"] = (
        len(rows),
        f'query="{rows[0][0][:80]}..." gold_doc="{rows[0][1][:80]}..."',
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({k: {"count": v[0], "example": v[1]} for k, v in stats.items()}, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
