"""
test_baselines.py — Tests for baseline context selection methods.

Run with:
    pytest tests/test_baselines.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from haluguard.baselines import (
    blend_scores,
    bm25_select,
    cosine_scores,
    cosine_select,
    edit_select,
    jaccard_select,
    full_context_select,
    gold_only_select,
    minmax_scores_to_unit,
    no_context_select,
    random_ranking,
    random_select,
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

    def test_cosine_scores_shape_matches_candidates(self) -> None:
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        chunk_embs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        scores = cosine_scores(query_emb, chunk_embs)

        assert scores.shape == (2,)
        assert scores[0] > scores[1]


class TestLexicalBaselines:
    def test_jaccard_prefers_token_overlap(self) -> None:
        contexts = [
            {"snippet": "def foo(x): return x + 1"},
            {"snippet": "class Bar: pass"},
        ]

        result = jaccard_select("foo(x)", contexts, top_k=1)

        assert result == [0]

    def test_edit_prefers_more_similar_token_sequence(self) -> None:
        contexts = [
            {"snippet": "foo(bar, baz)"},
            {"snippet": "qux(value)"},
        ]

        result = edit_select("foo(bar)", contexts, top_k=1)

        assert result == [0]


class TestRandomSelect:
    def test_seeded_random_select_is_deterministic(self) -> None:
        a = random_select(5, top_k=3, seed=7)
        b = random_select(5, top_k=3, seed=7)

        assert a == b

    def test_random_ranking_covers_all_indices(self) -> None:
        ranking = random_ranking(5, seed=7)

        assert sorted(ranking) == [0, 1, 2, 3, 4]


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


class TestMinmaxScoresToUnit:
    def test_maps_to_zero_one(self) -> None:
        out = minmax_scores_to_unit(np.array([0.0, 0.5, 1.0], dtype=np.float64))
        assert np.allclose(out, [0.0, 0.5, 1.0])

    def test_constant_scores(self) -> None:
        out = minmax_scores_to_unit(np.array([3.0, 3.0, 3.0], dtype=np.float64))
        assert np.allclose(out, [0.5, 0.5, 0.5])

    def test_empty(self) -> None:
        out = minmax_scores_to_unit(np.array([], dtype=np.float64))
        assert out.size == 0


class TestBlendScores:
    def test_weight_one_is_scores_a(self) -> None:
        a = np.array([0.0, 1.0, 0.5], dtype=np.float64)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        out = blend_scores(a, b, weight_a=1.0, normalize=False)
        assert np.allclose(out, a)

    def test_equal_weights_average_when_normalized(self) -> None:
        a = np.array([0.0, 2.0], dtype=np.float64)
        b = np.array([0.0, 1.0], dtype=np.float64)
        out = blend_scores(a, b, weight_a=0.5, normalize=True)
        assert out.shape == (2,)
        assert np.all(np.logical_and(out >= 0.0, out <= 1.0))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            blend_scores(np.array([1.0, 2.0]), np.array([1.0]), weight_a=0.5)
