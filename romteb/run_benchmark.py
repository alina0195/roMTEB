"""Run RoMTEB on a model.

The public CLI is ``romteb-eval`` (also ``python -m romteb``). This module
keeps ``run_romteb()`` and remains callable as ``romteb/run_benchmark.py``.

USAGE:
    romteb-eval --model intfloat/multilingual-e5-large --preset smoke --output /out
    python -m romteb --model /uploads/job-42 --preset full --output /out

    # GPU only (login node has no CUDA; encodes look hung):
    sbatch scripts/sbatch_smoke_e5.sh
    sbatch --job-name=romteb-gpu scripts/run_gpu_job.sh \\
        romteb/run_benchmark.py --model intfloat/multilingual-e5-large --loader auto

    # Default (SentenceTransformer loader + declared MODEL_PROMPTS):
    ./apptainer-exec-romteb.sh romteb/run_benchmark.py --model intfloat/multilingual-e5-large
    ./apptainer-exec-romteb.sh romteb/run_benchmark.py --model BAAI/bge-m3 --tasks RoSTS RoDTALLawsRetrieval

    # Official MTEB wrapper (E5, Qwen3, jina-v3):
    ./apptainer-exec-romteb.sh romteb/run_benchmark.py --model Qwen/Qwen3-Embedding-4B --loader mteb

    # Romanian BM25 lexical baseline:
    ./apptainer-exec-romteb.sh romteb/run_benchmark.py --model romteb/bm25s-ro --tasks WebFAQRetrieval

    # ModernBERT / Granite (broken flash_attn in container):
    ./apptainer-exec-romteb.sh romteb/run_benchmark.py \\
        --model ibm-granite/granite-embedding-97m-multilingual-r2 --no_flash_attn

Output: `<output_dir>/<model-slug>/<TaskName>/<split>.json` (MTEB default layout).
IR predictions (for bootstrap CI) go to `<output_dir>/predictions/<model-slug>/`.
Classification tasks with samples_per_label != 8 also write `<Task>.k8.json`.
Platform contract: `<output_dir>/summary.json`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def _flash_attn_disabled() -> bool:
    return (
        "--no_flash_attn" in sys.argv
        or os.environ.get("ROMTEB_DISABLE_FLASH_ATTN", "").lower() in {"1", "true", "yes"}
    )


def _disable_flash_attn() -> None:
    """Force transformers to skip flash-attn (use SDPA/eager instead).

    Needed when flash_attn is installed but its CUDA extension was built
    against a different PyTorch version than the container provides.
    """
    import transformers.utils as utils
    import transformers.utils.import_utils as import_utils

    disabled = lambda: False
    import_utils.is_flash_attn_2_available = disabled
    utils.is_flash_attn_2_available = disabled

    for name in (
        "is_flash_attn_3_available",
        "is_flash_attn_greater_or_equal_2_10",
        "is_flash_attn_greater_or_equal",
    ):
        fn = getattr(import_utils, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
        if fn is not None:
            stub = lambda *args, **kwargs: False
            setattr(import_utils, name, stub)
            if hasattr(utils, name):
                setattr(utils, name, stub)

    orig_available = import_utils._is_package_available

    def _pkg_available(name: str, *args, **kwargs):
        if name == "flash_attn":
            return False
        return orig_available(name, *args, **kwargs)

    import_utils._is_package_available = _pkg_available

    for mod_name in list(sys.modules):
        if mod_name == "flash_attn" or mod_name.startswith("flash_attn."):
            del sys.modules[mod_name]
        if "modernbert" in mod_name:
            del sys.modules[mod_name]


if _flash_attn_disabled():
    _disable_flash_attn()

import copy

import mteb
from sentence_transformers import SentenceTransformer

from romteb.benchmark import ROMTEB_TASKS
from romteb.bm25_ro import BM25_MODEL_NAME, LEXICAL_BASELINE_NAMES, load_bm25_ro
from romteb.eval_config import DEFAULT_SAMPLES_PER_LABEL, classification_shots
from romteb.model_prompts import prefer_mteb_loader, prompts_for

# mteb/transformers import can re-bind utils.is_flash_attn_2_available; patch again.
if _flash_attn_disabled():
    _disable_flash_attn()


def _filter_tasks(all_tasks, names):
    if not names:
        return all_tasks
    wanted = set(names)
    selected = [t for t in all_tasks if getattr(t.metadata, "name", "") in wanted]
    missing = wanted - {t.metadata.name for t in selected}
    if missing:
        print(f"[romteb] tasks not in RoMTEB: {sorted(missing)}", file=sys.stderr)
    return selected


def _relax_nemotron_transformers_pin() -> None:
    """MTEB's LlamaEmbedNemotron loader requires transformers==4.51.0 exactly.

    Cluster SIFs ship a nearby 4.5x build that already ran this model; accept
    anything >= 4.51.0 so ``--loader auto`` does not crash on import.
    """
    try:
        from packaging.version import Version
        from mteb.models.model_implementations import nvidia_models as nv
    except Exception:
        return
    required = Version("4.51.0")
    try:
        found = Version(nv.transformers_version)
    except Exception:
        return
    if found >= required and found != required:
        print(
            f"[romteb] relaxing LlamaEmbedNemotron transformers pin "
            f"({nv.transformers_version} >= 4.51.0)"
        )
        nv.transformers_version = "4.51.0"


def _native_max_seq_length(model) -> int | None:
    inner = getattr(model, "model", model)
    for attr in ("max_seq_length", "max_seq_len"):
        v = getattr(inner, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    meta = getattr(model, "mteb_model_meta", None)
    if meta is not None:
        v = getattr(meta, "max_tokens", None)
        if isinstance(v, (int, float)) and v and v == v:
            return int(v)
    return None


def _load_model(
    model_name_or_path: str,
    loader: str,
    *,
    no_flash_attn: bool = False,
    trust_remote_code: bool = True,
):
    """Load a model using the specified loader strategy.

    loader="st":   SentenceTransformer + MODEL_PROMPTS
    loader="mteb": mteb.get_model (API models, task-aware / instruct wrappers)
    loader="auto": official MTEB wrapper when declared, else ST, else mteb
    """
    if model_name_or_path in LEXICAL_BASELINE_NAMES:
        print(f"[romteb] loading Romanian BM25 baseline ({BM25_MODEL_NAME})")
        return load_bm25_ro()

    if no_flash_attn:
        print("[romteb] flash-attn disabled; using attn_implementation=sdpa")
        _disable_flash_attn()

    if loader == "auto" and prefer_mteb_loader(model_name_or_path):
        loader = "mteb"
        print(f"[romteb] auto-selecting loader=mteb for {model_name_or_path}")

    if not trust_remote_code:
        print("[romteb] trust_remote_code=False (untrusted upload)")

    st_kwargs: dict = {"trust_remote_code": trust_remote_code}
    if no_flash_attn:
        st_kwargs["model_kwargs"] = {"attn_implementation": "sdpa"}

    prompts = prompts_for(model_name_or_path)
    if prompts:
        print(f"[romteb] MODEL_PROMPTS for {model_name_or_path}: {prompts}")
    else:
        print(f"[romteb] no encode prefix for {model_name_or_path}")

    if loader == "mteb":
        print(f"[romteb] loading via mteb.get_model({model_name_or_path!r})")
        _relax_nemotron_transformers_pin()
        return mteb.get_model(model_name_or_path)

    from mteb.models import SentenceTransformerEncoderWrapper

    if loader == "st":
        print(f"[romteb] loading via SentenceTransformer({model_name_or_path!r})")
        st = SentenceTransformer(model_name_or_path, **st_kwargs)
        return SentenceTransformerEncoderWrapper(
            st, model_prompts=prompts or None
        )

    try:
        print(f"[romteb] trying SentenceTransformer({model_name_or_path!r}) ...")
        st = SentenceTransformer(model_name_or_path, **st_kwargs)
        return SentenceTransformerEncoderWrapper(
            st, model_prompts=prompts or None
        )
    except Exception as st_exc:
        print(f"[romteb] SentenceTransformer failed: {st_exc}", file=sys.stderr)
        print(f"[romteb] falling back to mteb.get_model({model_name_or_path!r})")
        return mteb.get_model(model_name_or_path)


def _latest_task_json(output_dir: str, task_name: str) -> Path | None:
    hits = [
        p
        for p in Path(output_dir).rglob(f"{task_name}.json")
        if p.name == f"{task_name}.json"
    ]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def _annotate_model_meta(output_dir: str, max_seq_length: int | None) -> None:
    if max_seq_length is None:
        return
    for path in Path(output_dir).rglob("model_meta.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("romteb_max_seq_length") == max_seq_length:
            continue
        meta["romteb_max_seq_length"] = max_seq_length
        if meta.get("max_tokens") is None:
            meta["max_tokens"] = max_seq_length
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _run_task(
    task,
    model,
    output_dir: str,
    overwrite_results: bool,
    prediction_folder: Path | None,
    encode_kwargs: dict | None = None,
):
    evaluation = mteb.MTEB(tasks=[task])
    kwargs = {
        "output_folder": output_dir,
        "overwrite_results": overwrite_results,
    }
    if encode_kwargs:
        kwargs["encode_kwargs"] = encode_kwargs
    task_type = getattr(getattr(task, "metadata", None), "type", "")
    if prediction_folder is not None and task_type in {"Retrieval", "Reranking"}:
        kwargs["prediction_folder"] = str(prediction_folder)
    evaluation.run(model, **kwargs)


def _run_k8_sidecar(
    task,
    model,
    output_dir: str,
    overwrite_results: bool,
    encode_kwargs: dict | None = None,
) -> None:
    name = getattr(getattr(task, "metadata", None), "name", "")
    if not name or not hasattr(task, "samples_per_label"):
        return
    if classification_shots(name) == DEFAULT_SAMPLES_PER_LABEL:
        return
    tuned_json = _latest_task_json(output_dir, name)
    if tuned_json is None:
        print(f"[romteb] skip k8 sidecar for {name}: no tuned JSON", file=sys.stderr)
        return
    original_path = tuned_json
    backup = tuned_json.with_name(tuned_json.name + ".tuned")
    shutil.copy2(tuned_json, backup)
    original_k = task.samples_per_label
    try:
        try:
            k8_task = copy.deepcopy(task)
        except Exception:
            k8_task = task
        k8_task.samples_per_label = DEFAULT_SAMPLES_PER_LABEL
        print(f"[romteb] dual-k: {name} samples_per_label={DEFAULT_SAMPLES_PER_LABEL}")
        _run_task(
            k8_task,
            model,
            output_dir,
            overwrite_results=True,
            prediction_folder=None,
            encode_kwargs=encode_kwargs,
        )
        k8_json = _latest_task_json(output_dir, name)
        if k8_json is not None:
            dest = k8_json.with_name(f"{name}.k8.json")
            shutil.move(str(k8_json), str(dest))
            print(f"[romteb] wrote {dest}")
    except Exception as exc:
        print(f"[romteb] k8 sidecar failed for {name}: {exc}", file=sys.stderr)
    finally:
        task.samples_per_label = original_k
        if backup.exists():
            shutil.move(str(backup), str(original_path))


def _require_cuda(model_name_or_path: str, allow_cpu: bool) -> None:
    """Dense encoders must run on a GPU node. Login-node CPU looks hung."""
    if model_name_or_path in LEXICAL_BASELINE_NAMES:
        print("[romteb] BM25 is lexical; CUDA not required")
        return
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"[romteb] torch is required: {exc}") from exc
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        print(f"[romteb] CUDA ok  devices={n}  gpu0={name}")
        return
    if allow_cpu:
        print("[romteb] WARNING: no CUDA; running on CPU (--allow_cpu)")
        return
    raise SystemExit(
        "[romteb] no CUDA device. Do not run run_benchmark.py on the login "
        "node (fep*). Submit a GPU job:\n"
        "  sbatch --job-name=romteb-gpu scripts/run_gpu_job.sh "
        "romteb/run_benchmark.py --model ...\n"
        "Or: sbatch scripts/sbatch_smoke_e5.sh\n"
        "Pass --allow_cpu only to debug."
    )


def run_romteb(
    model_name_or_path: str,
    output_dir: str = "results",
    tasks: list[str] | None = None,
    overwrite_results: bool = False,
    loader: str = "auto",
    no_flash_attn: bool = False,
    skip_k8: bool = False,
    allow_cpu: bool = False,
    batch_size: int | None = None,
    trust_remote_code: bool = True,
) -> dict:
    _require_cuda(model_name_or_path, allow_cpu)
    model = _load_model(
        model_name_or_path,
        loader,
        no_flash_attn=no_flash_attn,
        trust_remote_code=trust_remote_code,
    )
    max_len = _native_max_seq_length(model)
    if max_len is not None:
        print(f"[romteb] native max_seq_length={max_len} (no common cap)")
    else:
        print("[romteb] native max_seq_length=unknown")

    task_list = _filter_tasks(ROMTEB_TASKS, tasks)
    if not tasks:
        skipped = [
            t.metadata.name
            for t in task_list
            if getattr(getattr(t, "metadata", None), "type", "") == "Clustering"
        ]
        if skipped:
            print(f"[romteb] skipping clustering: {skipped}")
        task_list = [
            t
            for t in task_list
            if getattr(getattr(t, "metadata", None), "type", "") != "Clustering"
        ]
    if model_name_or_path in LEXICAL_BASELINE_NAMES:
        task_list = [
            t
            for t in task_list
            if getattr(getattr(t, "metadata", None), "type", "")
            in {"Retrieval", "Reranking"}
        ]
        print("[romteb] BM25 restricted to Retrieval + Reranking")
    if not task_list:
        raise SystemExit("No tasks selected.")

    print(f"[romteb] running {len(task_list)} tasks on {model_name_or_path}")
    for t in task_list:
        print(f"  - {t.metadata.name}  ({t.metadata.type})")

    encode_kwargs = {"batch_size": batch_size} if batch_size is not None else None
    if encode_kwargs:
        print(f"[romteb] encode batch_size={batch_size}")

    slug = model_name_or_path.replace("/", "__")
    prediction_folder = Path(output_dir) / "predictions" / slug
    prediction_folder.mkdir(parents=True, exist_ok=True)

    failed = []
    for task in task_list:
        task_name = getattr(task.metadata, "name", repr(task))
        try:
            _run_task(
                task,
                model,
                output_dir,
                overwrite_results,
                prediction_folder,
                encode_kwargs=encode_kwargs,
            )
            if not skip_k8:
                _run_k8_sidecar(
                    task,
                    model,
                    output_dir,
                    overwrite_results,
                    encode_kwargs=encode_kwargs,
                )
        except Exception as exc:
            print(f"[romteb] FAILED {task_name}: {exc}", file=sys.stderr)
            failed.append(task_name)

    _annotate_model_meta(output_dir, max_len)

    if failed:
        print(f"[romteb] {len(failed)} task(s) failed: {failed}", file=sys.stderr)
    return {
        "failed_tasks": failed,
        "n_tasks": len(task_list),
        "output_dir": output_dir,
    }


def main():
    from romteb.cli import main as cli_main

    raise SystemExit(cli_main())


if __name__ == "__main__":
    main()
