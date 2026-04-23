"""
benchmarks/crosscodeeval.py — CrossCodeEval (Python split) adapter.

By default the loader first tries the project benchmark id
``ArtifactAI/cceval`` and then falls back to ``Vincentvmt/CrossCodeEval`` for
local compatibility. Each example provides a ``prompt`` (code-so-far), a
``groundtruth`` next line, and ``crossfile_context`` — a dict whose ``list``
field contains candidate cross-file snippets.

CrossCodeEval does **not** provide a labelled gold chunk, so ``gold_index`` is
always ``None``. Retrieval metrics are therefore not reported for this
benchmark; only generation-quality metrics are.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from haluguard.benchmarks.base import Example, default_cache_dir


class CrossCodeEvalLoader:
    """Loader for the Python split of CrossCodeEval."""

    name = "crosscodeeval"

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config_name: Optional[str] = "python",
        split: str = "test",
        cache_dir: Optional[Path] = None,
        dataset: Optional[Any] = None,
    ) -> None:
        """Initialise the loader.

        Args:
            repo_id:     HuggingFace dataset identifier. If omitted, the loader
                         prefers ``ArtifactAI/cceval`` and then falls back to
                         ``Vincentvmt/CrossCodeEval``.
            config_name: Dataset configuration (language). Defaults to
                         ``"python"``. Set to ``None`` if the dataset has no
                         sub-configs.
            split:       Split name. Defaults to ``"test"``.
            cache_dir:   HuggingFace cache dir.
            dataset:     Pre-loaded iterable of raw examples (skips
                         ``load_dataset``).
        """
        self.repo_id = repo_id
        self.config_name = config_name
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self._dataset = dataset

    def _candidate_repo_ids(self) -> List[str]:
        """Return dataset ids to try in priority order."""
        if self.repo_id is not None:
            return [self.repo_id]

        candidates = []
        env_repo_id = os.environ.get("HALUGUARD_CCEVAL_REPO_ID")
        if env_repo_id:
            candidates.append(env_repo_id)
        candidates.extend(
            [
                "ArtifactAI/cceval",
                "Vincentvmt/CrossCodeEval",
            ]
        )

        seen = set()
        out: List[str] = []
        for repo_id in candidates:
            if repo_id not in seen:
                out.append(repo_id)
                seen.add(repo_id)
        return out

    def _load(self) -> Any:
        if self._dataset is not None:
            return self._dataset
        from datasets import load_dataset  # type: ignore

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs: Dict[str, Any] = {
            "split": self.split,
            "cache_dir": str(self.cache_dir),
        }
        errors: List[str] = []
        for repo_id in self._candidate_repo_ids():
            try:
                if self.config_name is not None:
                    ds = load_dataset(repo_id, self.config_name, **kwargs)
                else:
                    ds = load_dataset(repo_id, **kwargs)
                self.repo_id = repo_id
                self._dataset = ds
                return ds
            except Exception as exc:
                errors.append(f"{repo_id}: {type(exc).__name__}: {exc}")

        raise RuntimeError(
            "Unable to load CrossCodeEval from any configured dataset id:\n"
            + "\n".join(f"  - {msg}" for msg in errors)
        )

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


def _extract_chunks(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalise CrossCodeEval's ``crossfile_context`` into ``[{snippet,path}]``.

    The field can appear in multiple shapes across revisions:
    - ``{"list": [{"text": ..., "filename": ...}, ...]}``
    - ``{"list": [{"retrieved_chunk": ..., "filename": ...}, ...]}``
    - list-of-dicts directly at the top level.
    """
    raw = row.get("crossfile_context")
    if raw is None:
        raw = row.get("context") or row.get("cross_file_context")

    items: List[Any]
    if isinstance(raw, dict):
        items = raw.get("list") or raw.get("chunks") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    out: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        snippet = (
            item.get("retrieved_chunk")
            or item.get("text")
            or item.get("snippet")
            or item.get("content")
            or ""
        )
        path = (
            item.get("filename")
            or item.get("path")
            or item.get("file_path")
            or ""
        )
        if snippet:
            out.append({"snippet": str(snippet), "path": str(path)})
    return out


def _row_to_example(row: Dict[str, Any], idx: int) -> Optional[Example]:
    """Convert a raw CrossCodeEval row into the uniform :class:`Example`."""
    cropped_code = row.get("prompt") or row.get("cropped_code") or ""
    reference = row.get("groundtruth") or row.get("next_line") or ""
    if not cropped_code.strip() or not reference.strip():
        return None

    chunks = _extract_chunks(row)
    task_id = row.get("metadata", {}).get("task_id") if isinstance(row.get("metadata"), dict) else None
    task_id = task_id or f"cceval::{idx}"

    return Example(
        cropped_code=cropped_code,
        context_chunks=chunks,
        gold_index=None,
        reference=reference,
        language="python",
        import_statement="",
        metadata={
            "task_id": task_id,
            "source_index": idx,
        },
    )
