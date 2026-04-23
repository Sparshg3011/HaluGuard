"""
benchmarks/repobench.py — RepoBench v1.1 adapter.

Wraps ``tianyang/repobench_python_v1.1`` (split ``cross_file_first``) into the
common :class:`Example` shape. Each example already has pre-extracted context
chunks and a labelled ``gold_snippet_index``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from haluguard.benchmarks.base import Example, default_cache_dir


class RepoBenchLoader:
    """Loader for ``tianyang/repobench_python_v1.1`` (``cross_file_first``)."""

    name = "repobench"

    def __init__(
        self,
        split: str = "cross_file_first",
        cache_dir: Optional[Path] = None,
        dataset: Optional[Any] = None,
    ) -> None:
        """Initialise the loader.

        Args:
            split:     Dataset split to use. ``cross_file_first`` is the only
                       split HaluGuard is evaluated on.
            cache_dir: HuggingFace cache dir. Defaults to the project-wide
                       ``~/.cache/haluguard`` or the ``HALUGUARD_DATA_DIR`` env
                       override.
            dataset:   Pre-loaded iterable of RepoBench examples. When supplied,
                       ``load_dataset`` is skipped entirely (useful for tests
                       and for notebook runs that already loaded the split).
        """
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self._dataset = dataset

    def _load(self) -> Any:
        if self._dataset is not None:
            return self._dataset
        from datasets import load_dataset  # type: ignore

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(
            "tianyang/repobench_python_v1.1",
            split=self.split,
            cache_dir=str(self.cache_dir),
        )
        self._dataset = ds
        return ds

    def iter_examples(self, limit: Optional[int] = None) -> Iterator[Example]:
        """Yield :class:`Example` objects (optionally capped at ``limit``)."""
        ds = self._load()
        count = 0
        for idx, row in enumerate(ds):
            ex = _row_to_example(row, idx)
            if ex is None:
                continue
            yield ex
            count += 1
            if limit is not None and count >= limit:
                return


def _row_to_example(row: Dict[str, Any], idx: int) -> Optional[Example]:
    """Convert a raw RepoBench row into the uniform :class:`Example`."""
    context = row.get("context") or []
    reference = row.get("next_line") or row.get("gold_line") or ""
    cropped_code = row.get("cropped_code") or row.get("code") or ""

    if not cropped_code.strip() or not reference.strip():
        return None

    # Normalise chunks to {"snippet","path"} — source sometimes has "identifier"
    # and other extras.
    chunks: List[Dict[str, str]] = []
    for ch in context:
        snippet = ch.get("snippet", "")
        path = ch.get("path", "")
        chunks.append({"snippet": snippet, "path": path})

    gold_idx = row.get("gold_snippet_index")
    if gold_idx is not None and (gold_idx < 0 or gold_idx >= len(chunks)):
        gold_idx = None

    task_id = f"{row.get('repo_name', 'repo')}::{row.get('file_path', 'file')}::{idx}"

    return Example(
        cropped_code=cropped_code,
        context_chunks=chunks,
        gold_index=gold_idx,
        reference=reference,
        language="python",
        import_statement=row.get("import_statement", "") or "",
        metadata={
            "task_id": task_id,
            "repo_name": row.get("repo_name"),
            "file_path": row.get("file_path"),
            "source_index": idx,
        },
    )
