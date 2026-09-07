"""RoMTEB - Romanian Massive Text Embedding Benchmark."""

from __future__ import annotations

from typing import Any

__all__ = ["ROMTEB_TASKS", "get_benchmark", "write_summary", "run_romteb"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in {"ROMTEB_TASKS", "get_benchmark", "ROMTEB_CUSTOM", "ROMTEB_REUSED"}:
        from romteb import benchmark

        return getattr(benchmark, name)
    if name == "write_summary":
        from romteb.summary import write_summary

        return write_summary
    if name == "run_romteb":
        from romteb.run_benchmark import run_romteb

        return run_romteb
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
