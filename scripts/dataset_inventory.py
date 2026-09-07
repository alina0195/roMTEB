"""Collect RoMTEB dataset metadata, test-set sizes, and examples.

Usage:
    ./apptainer-exec-romteb.sh romteb/scripts/dataset_inventory.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

from romteb._hf_datasets import Dataset, load_dataset

from romteb.benchmark import ROMTEB_TASKS
from romteb.data_prep._classification_utils import stratified_train_test_split


def _truncate(text: str, limit: int = 100) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _dataset_path(meta) -> str:
    dataset = getattr(meta, "dataset", None) or {}
    if isinstance(dataset, dict):
        return dataset.get("path", "?")
    return getattr(dataset, "path", "?")


def _language_kind(task) -> str:
    meta = task.metadata
    task_type = str(meta.type)
    if task_type == "BitextMining":
        return "Multilingual (Romanian bitext pairs)"
    sample = (getattr(meta, "sample_creation", "") or "").lower()
    desc = (meta.description or "").lower()
    if any(k in sample or k in desc for k in ("machine-translated", "translated", "cross-lingual")):
        return "Romanian (translated/derived)"
    if meta.reference and any(
        x in meta.reference.lower()
        for x in ("massive", "sib200", "eurlex", "xquad", "flores", "ntrex", "iwslt", "tatoeba", "webfaq", "wikipedia")
    ):
        if task_type in {"Retrieval", "BitextMining", "Classification", "Clustering"}:
            if "romanian" in desc or "ron" in str(getattr(meta, "eval_langs", "")):
                return "Romanian subset of multilingual dataset"
    return "Romanian"


def _first_row(data):
    if data is None:
        return None
    if hasattr(data, "__getitem__") and hasattr(data, "column_names"):
        return data[0]
    if isinstance(data, dict) and data:
        first = next(iter(data.values()))
        if isinstance(first, dict):
            return first
        return {"value": first}
    return None


def _format_example(task, payload) -> str:
    task_type = str(task.metadata.type)
    row = _first_row(payload)
    if row is None:
        return "n/a"

    if task_type == "Classification":
        text = row.get("text") or row.get("sentence") or row.get("input") or "?"
        return f'text="{_truncate(text)}" label={row.get("label", "?")}'

    if task_type in {"PairClassification", "STS"}:
        s1 = row.get("sentence1") or row.get("text1") or "?"
        s2 = row.get("sentence2") or row.get("text2") or "?"
        score = row.get("label", row.get("score", "?"))
        return f's1="{_truncate(s1)}" s2="{_truncate(s2)}" score={score}'

    if task_type == "Retrieval":
        queries = payload.get("queries", {}) if isinstance(payload, dict) else {}
        corpus = payload.get("corpus", {}) if isinstance(payload, dict) else {}
        qrels = payload.get("relevant_docs", payload.get("qrels", {})) if isinstance(payload, dict) else {}
        if isinstance(queries, Dataset):
            qrow = queries[0]
            qtext = qrow.get("text", qrow)
            qid = qrow.get("_id", "0")
            rel = qrels.get(qid, {}) if isinstance(qrels, dict) else {}
            if rel:
                doc_id = next(iter(rel))
                doc = corpus[doc_id] if isinstance(corpus, dict) else None
                if isinstance(doc, dict):
                    dtext = doc.get("text", doc)
                    return f'query="{_truncate(qtext)}" gold_doc="{_truncate(dtext)}"'
            return f'query="{_truncate(qtext)}"'
        if isinstance(queries, dict) and queries:
            qid = next(iter(queries))
            qval = queries[qid]
            qtext = qval.get("text", qval) if isinstance(qval, dict) else qval
            rel = qrels.get(qid, {}) if isinstance(qrels, dict) else {}
            if rel:
                doc_id = next(iter(rel))
                doc = corpus.get(doc_id, {}) if isinstance(corpus, dict) else {}
                dtext = doc.get("text", doc) if isinstance(doc, dict) else doc
                return f'query="{_truncate(qtext)}" gold_doc="{_truncate(dtext)}"'
            return f'query="{_truncate(qtext)}"'
        return "n/a"

    if task_type == "BitextMining":
        s1 = row.get("sentence1") or row.get("text1") or "?"
        s2 = row.get("sentence2") or row.get("text2") or "?"
        return f's1="{_truncate(s1)}" s2="{_truncate(s2)}"'

    if task_type == "Clustering":
        text = row.get("text") or row.get("sentences") or "?"
        return f'text="{_truncate(text)}" label={row.get("label", "?")}'

    return _truncate(str(row))


def _count_payload(task, payload) -> int | str:
    task_type = str(task.metadata.type)
    if payload is None:
        return "?"

    if task_type == "Retrieval" and isinstance(payload, dict):
        queries = payload.get("queries")
        if isinstance(queries, Dataset):
            return len(queries)
        if isinstance(queries, dict):
            return len(queries)
        return "?"

    if hasattr(payload, "__len__"):
        return len(payload)
    if isinstance(payload, dict):
        return sum(len(v) for v in payload.values() if hasattr(v, "__len__"))
    return "?"


def _resolve_task_block(task):
    task.load_data()
    split = task.eval_splits[0]
    subset = task.hf_subsets[0] if getattr(task, "hf_subsets", None) else "default"
    task_type = str(task.metadata.type)

    if task_type == "Retrieval":
        if getattr(task, "queries", None) is not None:
            queries = task.queries[subset][split]
            corpus = task.corpus[subset][split]
            relevant = task.relevant_docs[subset][split]
            return {
                "queries": queries,
                "corpus": corpus,
                "relevant_docs": relevant,
            }

        block = task.dataset
        if subset in block:
            block = block[subset]
        if isinstance(block, dict) and split in block:
            block = block[split]
        if isinstance(block, dict):
            return block
        return {"queries": block}

    block = task.dataset
    if isinstance(block, dict):
        if subset in block:
            sub = block[subset]
            if isinstance(sub, dict) and split in sub:
                return sub[split]
            return sub
        if split in block:
            return block[split]
    return block


FALLBACK_SOURCES: dict[str, dict] = {
    "RoABSAClassification": {"path": "upb-nlp/RoABSA", "split": "train", "kind": "classification"},
    "RoOffenseClassification": {"path": "readerbench/news-ro-offense", "kind": "classification"},
    "HateSpeechROClassification": {"path": "readerbench/ro-hate-speech", "kind": "classification"},
    "RoSTS": {"path": "dumitrescustefan/ro_sts", "split": "test", "kind": "sts"},
    "MedQARoRetrieval": {"local_csv": "datasets/MedQARo Dataset/romedqa_test_dataset.csv", "kind": "medqa"},
    # XQuAD-ro is reused from MMTEB as XQuADRetrieval (custom duplicate removed).
    "XQuADRetrieval": {"path": "google/xquad", "name": "xquad.ro", "split": "validation", "kind": "xquad"},
}


def _load_fallback(name: str) -> tuple[int | str, str]:
    spec = FALLBACK_SOURCES[name]
    kind = spec["kind"]

    if kind == "medqa":
        path = Path(__file__).resolve().parents[2] / spec["local_csv"]
        rows = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get("Intrebare") or "").strip()
                c = (row.get("Epicriza") or "").strip()
                if q and c:
                    rows.append((q, c))
        if not rows:
            return "?", "n/a"
        q, c = rows[0]
        return len(rows), f'query="{_truncate(q)}" gold_doc="{_truncate(c)}"'

    if kind == "xquad":
        ds = load_dataset(spec["path"], spec["name"], split=spec["split"])
        row = ds[0]
        return len(ds), f'query="{_truncate(row["question"])}" gold_doc="{_truncate(row["context"])}"'

    if kind == "sts":
        ds = load_dataset(spec["path"], split=spec["split"])
        row = ds[0]
        return len(ds), (
            f's1="{_truncate(row.get("sentence1", row.get("sentence_a", "?")))}" '
            f's2="{_truncate(row.get("sentence2", row.get("sentence_b", "?")))}" '
            f'score={row.get("score", row.get("label", "?"))}'
        )

    raw = load_dataset(spec["path"])
    base = raw[spec.get("split", "train")] if spec.get("split") in raw else raw[list(raw.keys())[0]]
    rows = []
    for r in base:
        text = r.get("text") or r.get("sentence") or r.get("comment") or r.get("Tweet") or ""
        label = r.get("label")
        if text and label is not None:
            rows.append({"text": str(text).strip(), "label": label})
    if not rows:
        return "?", "n/a"
    test = stratified_train_test_split(Dataset.from_list(rows))["test"]
    ex = test[0]
    return len(test), f'text="{_truncate(ex["text"])}" label={ex["label"]}'


def inspect_task(task) -> dict:
    meta = task.metadata
    row = {
        "task_name": meta.name,
        "task_type": str(meta.type),
        "source": meta.reference or _dataset_path(meta),
        "hf_path": _dataset_path(meta),
        "language": _language_kind(task),
        "test_instances": "?",
        "example": "n/a",
        "notes": "",
        "error": "",
    }
    try:
        payload = _resolve_task_block(task)
        row["test_instances"] = _count_payload(task, payload)
        row["example"] = _format_example(task, payload)
        if row["test_instances"] == 0 and meta.name in FALLBACK_SOURCES:
            count, example = _load_fallback(meta.name)
            row["test_instances"] = count
            row["example"] = example
            row["notes"] = "count from original source (HF mirror empty/unavailable)"
    except Exception as exc:
        if meta.name in FALLBACK_SOURCES:
            try:
                count, example = _load_fallback(meta.name)
                row["test_instances"] = count
                row["example"] = example
                row["notes"] = "loaded from original source"
            except Exception as fb_exc:
                row["error"] = f"{exc}; fallback: {fb_exc}"
        else:
            row["error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "docs" / "datasets_overview.md"),
    )
    args = parser.parse_args()

    rows = [inspect_task(task) for task in ROMTEB_TASKS]
    rows.sort(key=lambda r: (r["task_type"], r["task_name"]))

    lines = [
        "# RoMTEB benchmark datasets",
        "",
        f"Overview of all **{len(rows)}** tasks registered in `romteb/benchmark.py`.",
        "Test-set sizes and examples were collected with `scripts/dataset_inventory.py`.",
        "",
        "| Task | Type | Source | Language | Test instances | Example |",
        "|---|---|---|---|---:|---|",
    ]
    for r in rows:
        source = r["source"] or ""
        if r["hf_path"] and r["hf_path"] != "?":
            source = f"{source} (`{r['hf_path']}`)" if source else f"`{r['hf_path']}`"
        example = (r["example"] if not r["error"] else f"ERROR: {r['error']}").replace("|", "\\|")
        count = r["test_instances"]
        if r["notes"]:
            count = f"{count}*"
        lines.append(
            f"| `{r['task_name']}` | {r['task_type']} | {source} | {r['language']} | "
            f"{count} | {example} |"
        )

    notes = [r for r in rows if r["notes"]]
    errors = [r for r in rows if r["error"]]
    if notes or errors:
        lines.extend(["", "## Notes", ""])
    for r in notes:
        lines.append(f"- * `{r['task_name']}`: {r['notes']}.")
    if errors:
        lines.append("")
        lines.append("### Load errors")
        lines.append("")
        for r in errors:
            lines.append(f"- `{r['task_name']}`: {r['error']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    if errors:
        print(f"{len(errors)} task(s) still failed.", file=sys.stderr)


if __name__ == "__main__":
    main()
