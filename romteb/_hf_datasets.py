"""Import HuggingFace `datasets` despite the local `datasets/` data directory.

The workspace contains a `datasets/` folder (raw corpora). With
`PYTHONPATH=<workspace>`, `import datasets` would bind that folder instead of
the Hugging Face package. Every prep script and `_common.py` must import
through this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent


def _load_hf_datasets():
    skip = {_WORKSPACE.resolve()}
    orig = list(sys.path)
    filtered: list[str] = []
    for p in orig:
        try:
            if Path(p).resolve() in skip:
                continue
        except OSError:
            pass
        filtered.append(p)
    sys.path[:] = filtered
    try:
        import datasets as _ds
        path = getattr(_ds, "__file__", None)
        if path and Path(path).resolve() == (_WORKSPACE / "datasets").resolve():
            raise ImportError(
                "HuggingFace `datasets` is still shadowed by the local datasets/ folder"
            )
        return _ds
    finally:
        sys.path[:] = orig


_ds = _load_hf_datasets()
Dataset = _ds.Dataset
DatasetDict = _ds.DatasetDict
load_dataset = _ds.load_dataset
concatenate_datasets = _ds.concatenate_datasets
Features = getattr(_ds, "Features", None)
Value = getattr(_ds, "Value", None)
ClassLabel = getattr(_ds, "ClassLabel", None)
