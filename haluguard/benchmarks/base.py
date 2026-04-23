"""
benchmarks/base.py — Common types for benchmark adapters.

Every loader yields :class:`Example` objects with a uniform shape so the
runner in :mod:`haluguard.run_eval` can iterate over any benchmark without
per-source special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable


@dataclass
class Example:
    """A single benchmark example with cross-file context.

    Attributes:
        cropped_code:     Code written so far in the current file (query).
        context_chunks:   List of dicts with ``"snippet"`` and ``"path"`` keys.
                          Each chunk is a candidate cross-file context.
        gold_index:       Index of the gold chunk in ``context_chunks``, or
                          ``None`` if the benchmark does not label it.
        reference:        Ground-truth next line to be predicted.
        language:         Programming language (``"python"`` for both current
                          benchmarks).
        import_statement: Optional import block from the current file (used by
                          RepoBench; may be empty for benchmarks that fold it
                          into ``cropped_code``).
        metadata:         Free-form dict with benchmark-specific extras (repo
                          name, task_id, etc.).
    """

    cropped_code: str
    context_chunks: List[Dict[str, str]]
    gold_index: Optional[int]
    reference: str
    language: str = "python"
    import_statement: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BenchmarkLoader(Protocol):
    """Protocol every benchmark adapter must satisfy."""

    name: str

    def iter_examples(self, limit: Optional[int] = None) -> Iterator[Example]:
        """Yield :class:`Example` objects from the benchmark.

        Args:
            limit: If not ``None``, stop after yielding this many examples.
        """
        ...


def default_cache_dir() -> Path:
    """Project-wide default cache root (``~/.cache/haluguard`` or the env override)."""
    import os

    override = os.environ.get("HALUGUARD_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "haluguard"
