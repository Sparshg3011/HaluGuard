"""
test_baselines.py — Tests for baseline context selection methods.

Run with:
    pytest tests/test_baselines.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from haluguard.baselines import (
    bm25_select,
    cosine_select,
    full_context_select,
    gold_only_select,
    no_context_select,
)


class TestBM25Select:
    def test_returns_correct_count(self) -> None:
        contexts = [
            {"snippet": "def foo(): return 1"},
            {"snippet": "def bar(): return 2"},
            {"snippet": "def baz(): return 3"},
        ]
        result = bm25_select("return foo()", contexts, top_k=2)
        assert len(result) == 2

    def test_top_k_larger_than_contexts(self) -> None:
        contexts = [{"snippet": "x = 1"}, {"snippet": "y = 2"}]
        result = bm25_select("x", contexts, top_k=10)
        assert len(result) == 2

    def test_empty_contexts(self) -> None:
        assert bm25_select("query", [], top_k=5) == []

    def test_returns_indices(self) -> None:
        contexts = [
            {"snippet": "import os"},
            {"snippet": "import sys"},
            {"snippet": "completely unrelated words"},
        ]
        result = bm25_select("import os", contexts, top_k=1)
        assert len(result) == 1
        assert isinstance(result[0], int)


class TestCosineSelect:
    def test_returns_correct_count(self) -> None:
        query_emb = np.random.randn(768).astype(np.float32)
        chunk_embs = np.random.randn(5, 768).astype(np.float32)
        result = cosine_select(query_emb, chunk_embs, top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_chunks(self) -> None:
        query_emb = np.random.randn(768).astype(np.float32)
        chunk_embs = np.random.randn(2, 768).astype(np.float32)
        result = cosine_select(query_emb, chunk_embs, top_k=10)
        assert len(result) == 2

    def test_empty_chunks(self) -> None:
        query_emb = np.random.randn(768).astype(np.float32)
        chunk_embs = np.empty((0, 768), dtype=np.float32)
        assert cosine_select(query_emb, chunk_embs) == []

    def test_most_similar_ranked_first(self) -> None:
        query_emb = np.array([1.0, 0.0, 0.0])
        chunk_embs = np.array([
            [0.0, 1.0, 0.0],   # orthogonal
            [0.9, 0.1, 0.0],   # very similar
            [0.5, 0.5, 0.0],   # somewhat similar
        ])
        result = cosine_select(query_emb, chunk_embs, top_k=3)
        assert result[0] == 1  # most similar first


class TestNoContextSelect:
    def test_returns_empty(self) -> None:
        assert no_context_select() == []


class TestFullContextSelect:
    def test_returns_all_indices(self) -> None:
        assert full_context_select(5) == [0, 1, 2, 3, 4]

    def test_zero_chunks(self) -> None:
        assert full_context_select(0) == []


class TestGoldOnlySelect:
    def test_returns_single_index(self) -> None:
        assert gold_only_select(3) == [3]

    def test_returns_zero_index(self) -> None:
        assert gold_only_select(0) == [0]
