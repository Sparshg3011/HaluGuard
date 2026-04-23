"""
test_benchmarks.py — Loader adapters produce well-formed Example objects.

Uses tiny in-memory fixtures to avoid any HuggingFace network dependency.
"""

from __future__ import annotations

from typing import Any, Dict, List

from haluguard.benchmarks.base import BenchmarkLoader, Example
from haluguard.benchmarks.crosscodeeval import CrossCodeEvalLoader
from haluguard.benchmarks.repobench import RepoBenchLoader


def _fake_repobench_row() -> Dict[str, Any]:
    return {
        "repo_name": "demo/repo",
        "file_path": "pkg/mod.py",
        "cropped_code": "def foo(x):\n    return",
        "next_line": "    return x + 1",
        "import_statement": "import os",
        "context": [
            {"snippet": "def helper(): ...", "path": "pkg/util.py"},
            {"snippet": "class Cfg: ...", "path": "pkg/cfg.py"},
        ],
        "gold_snippet_index": 0,
    }


def _fake_cceval_row() -> Dict[str, Any]:
    return {
        "prompt": "def compute(x):\n    return",
        "groundtruth": "    return x * 2",
        "crossfile_context": {
            "list": [
                {"retrieved_chunk": "def mult(a, b): ...", "filename": "ops.py"},
                {"text": "class Base: ...", "filename": "base.py"},
            ]
        },
        "metadata": {"task_id": "proj/task-1"},
    }


class TestRepoBenchLoader:
    def test_yields_examples(self) -> None:
        loader = RepoBenchLoader(dataset=[_fake_repobench_row() for _ in range(3)])
        assert loader.name == "repobench"
        assert isinstance(loader, BenchmarkLoader)
        examples = list(loader.iter_examples())
        assert len(examples) == 3
        ex = examples[0]
        assert isinstance(ex, Example)
        assert ex.cropped_code.startswith("def foo")
        assert ex.reference == "    return x + 1"
        assert ex.gold_index == 0
        assert ex.language == "python"
        assert ex.import_statement == "import os"
        assert len(ex.context_chunks) == 2
        assert set(ex.context_chunks[0].keys()) >= {"snippet", "path"}
        assert ex.metadata["task_id"].startswith("demo/repo")

    def test_limit(self) -> None:
        loader = RepoBenchLoader(dataset=[_fake_repobench_row() for _ in range(5)])
        examples = list(loader.iter_examples(limit=2))
        assert len(examples) == 2

    def test_skips_invalid_gold_index(self) -> None:
        bad = _fake_repobench_row()
        bad["gold_snippet_index"] = 99
        loader = RepoBenchLoader(dataset=[bad])
        ex = next(loader.iter_examples())
        # Out-of-range gold index is blanked, not propagated.
        assert ex.gold_index is None


class TestCrossCodeEvalLoader:
    def test_yields_examples(self) -> None:
        loader = CrossCodeEvalLoader(dataset=[_fake_cceval_row() for _ in range(2)])
        assert loader.name == "crosscodeeval"
        assert isinstance(loader, BenchmarkLoader)
        examples = list(loader.iter_examples())
        assert len(examples) == 2
        ex = examples[0]
        assert ex.reference == "    return x * 2"
        assert ex.gold_index is None  # CCEval has no labelled gold
        assert len(ex.context_chunks) == 2
        assert ex.context_chunks[0]["snippet"].startswith("def mult")
        assert ex.context_chunks[0]["path"] == "ops.py"
        assert ex.metadata["task_id"] == "proj/task-1"

    def test_skips_empty_rows(self) -> None:
        rows: List[Dict[str, Any]] = [
            {"prompt": "", "groundtruth": "x"},
            _fake_cceval_row(),
        ]
        loader = CrossCodeEvalLoader(dataset=rows)
        assert len(list(loader.iter_examples())) == 1
