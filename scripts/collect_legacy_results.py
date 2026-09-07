"""Build ``legacy/``: current-roster models on the v1 task protocol.

Pulls from ``results/`` first, then ``results_new/`` if a listed model is
missing. Drops ro-retriever / robert-retriever and other retired embedders.
Clustering JSON is not copied.

USAGE:
    ./apptainer-exec-romteb.sh scripts/collect_legacy_results.py
    ./apptainer-exec-romteb.sh scripts/make_plots.sh legacy
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Current eval roster (no retrievers). Slugs = HF id with "/" -> "__".
ROSTER = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Alibaba-NLP/gte-multilingual-base",
    "ibm-granite/granite-embedding-311m-multilingual-r2",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "google/embeddinggemma-300m",
    "Qwen/Qwen3-Embedding-0.6B",
    "Snowflake/snowflake-arctic-embed-m-v2.0",
    "nomic-ai/nomic-embed-text-v2-moe",
    "jinaai/jina-embeddings-v3",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    "intfloat/multilingual-e5-large-instruct",
    "Qwen/Qwen3-Embedding-8B",
    "BAAI/bge-multilingual-gemma2",
    # Base checkpoint for the RO MS MARCO fine-tune (no query/passage prefix).
    "BAAI/bge-m3",
    "alina0195/bge-m3-ro-msmarco-v2",
    # 8B/12B: default encode batch=32 OOM on H200 (llama: SaRoCo;
    # KaLM: WebFAQ, SciTechBan, SaRoCo, RoDTAL). Retry missing tasks only:
    #   --loader st --batch_size 1 --output results --tasks …
    "nvidia/llama-embed-nemotron-8b",
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    # Lexical baseline. Only produces scores on Retrieval + Reranking; the
    # rest stay empty. Needed to flag trivially lexical tasks (XQuAD,
    # MQARoCQA) and to detect dense models with wrong query prompts (any
    # dense model below BM25 on retrieval → suspect prompt).
    "romteb/bm25s-ro",
]

DROP_TASKS = {
    "SIB200ClusteringS2S",
    "RoNewsOutletClusteringP2P",
    "RoNewsTypeClusteringP2P",
    "JuRoLegalExamPairClassification",
    "WWTBMRoQAPairClassification",
    "RoMedQAv2PairClassification",
    "RoDTALLawsPairClassification",
    "Moroco.v2",
    "Moroco",
    # Retired in Sep 2026 (docs/task_audit.md): full-option-bank MCQ retrieval
    # variants collapsed to near-zero nDCG@10 because the corpus has duplicate
    # surface forms. Use *Reranking counterparts instead.
    "GrileGrammarRetrieval",
    "JuRoLegalExamRetrieval",
    "RoMedQAv2Retrieval",
    "WWTBMRoQARetrieval",
}


def slug(model: str) -> str:
    return model.replace("/", "__")


def _find_model_dir(slug_name: str, sources: list[Path]) -> Path | None:
    for src in sources:
        cand = src / slug_name
        if cand.is_dir():
            return cand
    return None


def collect(out_dir: Path, sources: list[Path], copy: bool) -> list[str]:
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir() and child.name not in {"plots"}:
                shutil.rmtree(child)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    lines = ["RoMTEB legacy snapshot (current roster, v1 tasks, no retrievers)", ""]
    present: list[str] = []
    for model in ROSTER:
        s = slug(model)
        src = _find_model_dir(s, sources)
        dest = out_dir / s
        if src is None:
            lines.append(f"MISSING  {model}")
            print(f"[legacy] missing {model}")
            continue
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        if copy:
            shutil.copytree(
                src,
                dest,
                ignore=shutil.ignore_patterns(
                    "*Clustering*.json",
                    *[f"{t}.json" for t in DROP_TASKS],
                    *[f"{t}.k8.json" for t in DROP_TASKS],
                ),
                symlinks=True,
            )
        else:
            dest.symlink_to(src.resolve())
        n_json = sum(1 for p in dest.rglob("*.json") if p.name.endswith(".json"))
        origin = src.relative_to(ROOT) if src.is_relative_to(ROOT) else src
        lines.append(f"OK       {model}  <- {origin}  ({n_json} json)")
        print(f"[legacy] {model} <- {origin}")
        present.append(model)

    ir_src = ROOT / "results" / "romteb_ir_task_stats.csv"
    if ir_src.exists():
        shutil.copy2(ir_src, out_dir / "romteb_ir_task_stats.csv")

    lines.append("")
    lines.append(f"n_present={len(present)}/{len(ROSTER)}")
    (out_dir / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return present


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(ROOT / "legacy"))
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy JSON trees instead of symlinking (frozen snapshot).",
    )
    args = parser.parse_args()
    sources = [ROOT / "results", ROOT / "results_new"]
    present = collect(Path(args.out_dir), sources, copy=args.copy)
    if not present:
        raise SystemExit("no roster models found under results/ or results_new/")
    print(f"Wrote {args.out_dir} ({len(present)} models)")


if __name__ == "__main__":
    main()
