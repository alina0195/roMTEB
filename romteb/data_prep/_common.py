"""Shared helpers for data-prep scripts.

All `prep_*.py` use these to (a) print a uniform stats summary and
(b) push to the HF Hub under `alina0195/romteb-<name>` while capturing
the resulting commit SHA for `TaskMetadata.dataset.revision` pinning.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from romteb._hf_datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


REVISIONS_FILE = Path(__file__).resolve().parent.parent.parent / "tasks" / "_revisions.json"
_PACKAGE_REVISIONS = Path(__file__).resolve().parent.parent / "tasks" / "_revisions.json"


def _revision_paths() -> list[Path]:
    env = os.environ.get("ROMTEB_REVISIONS")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.append(_PACKAGE_REVISIONS)
    paths.append(REVISIONS_FILE)
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _revisions_file() -> Path:
    for path in _revision_paths():
        if path.exists():
            return path
    return _revision_paths()[0]


def print_stats(name: str, ds: Dataset | DatasetDict, text_keys: list[str] | None = None) -> None:
    """Print row counts, label distribution (if `label` present), and text-length stats."""
    print(f"\n=== {name} ===")
    if isinstance(ds, DatasetDict):
        for split, sub in ds.items():
            _print_split_stats(f"{split}", sub, text_keys)
    else:
        _print_split_stats("data", ds, text_keys)


def _print_split_stats(label: str, ds: Dataset, text_keys: list[str] | None) -> None:
    print(f"  [{label}] rows={len(ds):,}  columns={ds.column_names}")
    if "label" in ds.column_names:
        dist = Counter(ds["label"])
        print(f"    label distribution: {dict(sorted(dist.items()))}")
    if text_keys:
        for k in text_keys:
            if k in ds.column_names:
                lengths = [len(s) for s in ds[k] if s is not None]
                if lengths:
                    print(
                        f"    {k}: min={min(lengths)} max={max(lengths)} "
                        f"mean={mean(lengths):.1f} chars"
                    )


def push_to_hub(
    ds: Dataset | DatasetDict,
    repo_id: str,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """Push dataset to HF Hub and return the resulting commit SHA.

    Records the SHA in `romteb/tasks/_revisions.json` so the task class can
    reference it later via `read_revision(repo_id)`.
    """
    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)

    commit_info = ds.push_to_hub(
        repo_id,
        private=private,
        commit_message=commit_message or f"Upload {repo_id}",
    )
    sha = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_sha", None)
    if sha is None:
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
        sha = refs.branches[0].target_commit if refs.branches else "main"
    _record_revision(repo_id, sha)
    print(f"Pushed: https://huggingface.co/datasets/{repo_id}  (rev {sha})")
    return sha


def _record_revision(repo_id: str, sha: str) -> None:
    targets = [p for p in _revision_paths() if p.exists()]
    if not targets:
        targets = [_revisions_file()]
    data: dict[str, Any] = {}
    newest = _revisions_file()
    if newest.exists():
        try:
            data = json.loads(newest.read_text())
        except json.JSONDecodeError:
            data = {}
    data[repo_id] = sha
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)


def read_revision(repo_id: str, default: str = "main") -> str:
    """Return the pinned commit SHA for `repo_id`, or `default` if unknown."""
    env_override = os.environ.get(f"ROMTEB_REV__{repo_id.replace('/', '__')}")
    if env_override:
        return env_override
    path = _revisions_file()
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text()).get(repo_id, default)
    except json.JSONDecodeError:
        return default
