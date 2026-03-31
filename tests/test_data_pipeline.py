"""
test_data_pipeline.py — Tests for RepoBench triplet generation and serialisation.

Run with:
    pytest tests/test_data_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from haluguard.data_pipeline import (
    ContrastiveTriplet,
    create_all_triplets,
    create_triplets_from_example,
    load_triplets,
    save_triplets,
    summarise_triplets,
)


def _make_example(
    n_contexts: int = 4,
    gold_idx: int = 1,
) -> Dict[str, Any]:
    """Build a synthetic RepoBench example for testing."""
    return {
        "repo_name": "test/repo",
        "file_path": "src/main.py",
        "cropped_code": "from utils import helper\nx = helper(",
        "context": [
            {
                "identifier": f"func_{i}",
                "path": f"src/module_{i}.py",
                "snippet": f"def func_{i}(arg):\n    return arg + {i}",
            }
            for i in range(n_contexts)
        ],
        "gold_snippet_index": gold_idx,
        "import_statement": "from utils import helper",
        "next_line": "x = helper(42)",
        "all_code": "from utils import helper\nx = helper(42)\nprint(x)",
        "created_at": "2023-01-01",
        "level": "2k",
    }


class TestCreateTripletsFromExample:
    def test_basic_triplet_count(self) -> None:
        example = _make_example(n_contexts=4, gold_idx=1)
        triplets = create_triplets_from_example(example, idx=0)
        # 4 contexts, 1 gold → 3 negatives → 3 triplets
        assert len(triplets) == 3

    def test_triplet_content(self) -> None:
        example = _make_example(n_contexts=3, gold_idx=0)
        triplets = create_triplets_from_example(example, idx=5)

        for t in triplets:
            assert t.query == example["cropped_code"]
            assert t.positive_context == example["context"][0]["snippet"]
            assert t.positive_path == example["context"][0]["path"]
            assert t.gold_snippet_index == 0
            assert "test/repo" in t.task_id
            assert "5" in t.task_id

    def test_invalid_gold_index_returns_empty(self) -> None:
        example = _make_example(n_contexts=3, gold_idx=99)
        assert create_triplets_from_example(example, idx=0) == []

    def test_negative_gold_index_returns_empty(self) -> None:
        example = _make_example(n_contexts=3, gold_idx=-1)
        assert create_triplets_from_example(example, idx=0) == []

    def test_single_context_returns_empty(self) -> None:
        example = _make_example(n_contexts=1, gold_idx=0)
        assert create_triplets_from_example(example, idx=0) == []

    def test_max_negatives_caps_output(self) -> None:
        example = _make_example(n_contexts=10, gold_idx=0)
        triplets = create_triplets_from_example(
            example, idx=0, max_negatives=3
        )
        assert len(triplets) == 3

    def test_max_negatives_larger_than_available(self) -> None:
        example = _make_example(n_contexts=4, gold_idx=0)
        triplets = create_triplets_from_example(
            example, idx=0, max_negatives=100
        )
        assert len(triplets) == 3  # only 3 negatives available


class TestCreateAllTriplets:
    def test_multiple_examples(self) -> None:
        dataset = [
            _make_example(n_contexts=3, gold_idx=0),
            _make_example(n_contexts=4, gold_idx=2),
        ]
        triplets = create_all_triplets(dataset)
        # 2 negatives from first + 3 negatives from second = 5
        assert len(triplets) == 5


class TestSerialisation:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        triplets = [
            ContrastiveTriplet(
                query="x = foo(",
                positive_context="def foo(): return 1",
                negative_context="def bar(): return 2",
                positive_path="foo.py",
                negative_path="bar.py",
                task_id="repo::file::0",
                gold_snippet_index=0,
            ),
            ContrastiveTriplet(
                query="y = bar(",
                positive_context="def bar(): return 2",
                negative_context="def baz(): return 3",
                positive_path="bar.py",
                negative_path="baz.py",
                task_id="repo::file::1",
                gold_snippet_index=1,
            ),
        ]
        path = tmp_path / "triplets.jsonl"
        save_triplets(triplets, path)

        loaded = load_triplets(path)
        assert len(loaded) == 2
        assert loaded[0].query == "x = foo("
        assert loaded[1].positive_path == "bar.py"
        assert loaded[0].gold_snippet_index == 0

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "triplets.jsonl"
        save_triplets([], path)
        assert path.exists()


class TestSummariseTriplets:
    def test_summary_stats(self) -> None:
        example = _make_example(n_contexts=5, gold_idx=0)
        triplets = create_triplets_from_example(example, idx=0)
        summary = summarise_triplets(triplets)
        assert summary["total"] == 4
        assert summary["unique_tasks"] == 1
        assert summary["avg_negatives_per_task"] == 4.0

    def test_empty_summary(self) -> None:
        summary = summarise_triplets([])
        assert summary["total"] == 0
        assert summary["unique_tasks"] == 0
