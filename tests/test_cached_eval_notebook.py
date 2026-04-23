"""Static checks for the official cached evaluation notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "05_cached_generation_evaluation.ipynb"


def _load_notebook() -> Dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell_sources(notebook: Dict[str, object]) -> List[str]:
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def test_cached_eval_notebook_is_cache_only() -> None:
    sources = "\n".join(_cell_sources(_load_notebook()))
    assert "compute_and_save_cceval_embeddings" not in sources
    assert "Pre-computing CrossCodeEval embeddings" not in sources
    assert "strictly cached-only" in sources
