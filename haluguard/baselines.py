"""
baselines.py — Baseline context selection methods for RepoBench evaluation.

Each function selects context chunk indices from the candidate list using
a different strategy.  These are compared against HCCS in the ablation study.

Baselines:
    - No context:           empty selection (lower bound)
    - BM25 top-k:           keyword overlap between cropped_code and snippets
    - CodeBERT cosine top-k: cosine similarity in embedding space
    - Full context:         all chunks (upper bound, may exceed context window)
    - Gold only (oracle):   always selects the gold chunk (theoretical ceiling)
"""

from __future__ import annotations

import random
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import numpy as np


def _tokenize_code(text: str) -> List[str]:
    """Tokenise code into identifiers, numbers, and punctuation."""
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\w\s]", text)


def _select_top_k_from_scores(
    scores: np.ndarray,
    top_k: int,
) -> List[int]:
    """Return the top-k indices for a descending score array."""
    if scores.shape[0] == 0:
        return []

    k = min(int(top_k), int(scores.shape[0]))
    return np.argsort(scores)[::-1][:k].tolist()


def bm25_select(
    cropped_code: str,
    contexts: List[Dict[str, str]],
    top_k: int = 5,
) -> List[int]:
    """Select context chunks by BM25 keyword overlap with the query.

    Uses ``rank_bm25`` to score each snippet against the tokenised query.

    Args:
        cropped_code: The code written so far (the query).
        contexts:     List of context dicts, each with at least a ``"snippet"`` key.
        top_k:        Number of top-scoring chunks to return.

    Returns:
        List of indices into ``contexts``, sorted by descending BM25 score.
        Length is ``min(top_k, len(contexts))``.
    """
    from rank_bm25 import BM25Okapi

    if not contexts:
        return []

    corpus = [c["snippet"].lower().split() for c in contexts]
    bm25 = BM25Okapi(corpus)
    query_tokens = cropped_code.lower().split()
    scores = bm25.get_scores(query_tokens)

    return _select_top_k_from_scores(np.asarray(scores, dtype=np.float64), top_k)


def jaccard_scores(
    cropped_code: str,
    contexts: List[Dict[str, str]],
) -> np.ndarray:
    """Score snippets using token-level Jaccard similarity."""
    query_tokens = set(_tokenize_code(cropped_code))
    scores: List[float] = []

    for context in contexts:
        snippet_tokens = set(_tokenize_code(context["snippet"]))
        union = query_tokens | snippet_tokens
        if not union:
            scores.append(0.0)
            continue
        scores.append(len(query_tokens & snippet_tokens) / len(union))

    return np.asarray(scores, dtype=np.float64)


def jaccard_select(
    cropped_code: str,
    contexts: List[Dict[str, str]],
    top_k: int = 5,
) -> List[int]:
    """Select context chunks by token-level Jaccard similarity."""
    return _select_top_k_from_scores(
        jaccard_scores(cropped_code, contexts),
        top_k=top_k,
    )


def edit_similarity_scores(
    cropped_code: str,
    contexts: List[Dict[str, str]],
) -> np.ndarray:
    """Score snippets using token-level edit similarity."""
    query_tokens = _tokenize_code(cropped_code)
    scores: List[float] = []

    for context in contexts:
        snippet_tokens = _tokenize_code(context["snippet"])
        scores.append(SequenceMatcher(a=query_tokens, b=snippet_tokens).ratio())

    return np.asarray(scores, dtype=np.float64)


def edit_select(
    cropped_code: str,
    contexts: List[Dict[str, str]],
    top_k: int = 5,
) -> List[int]:
    """Select context chunks by token-level edit similarity."""
    return _select_top_k_from_scores(
        edit_similarity_scores(cropped_code, contexts),
        top_k=top_k,
    )


def random_select(
    n_chunks: int,
    top_k: int = 5,
    seed: Optional[int] = None,
) -> List[int]:
    """Select chunks using a deterministic random permutation."""
    return random_ranking(n_chunks, seed=seed)[: min(int(top_k), int(n_chunks))]


def random_ranking(
    n_chunks: int,
    seed: Optional[int] = None,
) -> List[int]:
    """Return a deterministic random ranking over all chunk indices."""
    rng = random.Random(seed)
    shuffled = list(range(int(n_chunks)))
    rng.shuffle(shuffled)
    return shuffled


def minmax_scores_to_unit(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Linearly map a 1-D score vector to approximately ``[0, 1]``.

    Used when blending two scorers whose raw scales differ (e.g. HCCS sigmoid
    logits vs cosine similarities).

    Args:
        scores: 1-D array of per-candidate scores for one example.
        eps:    Numerical stability when all scores are equal.

    Returns:
        Same shape as ``scores``, values in ``[0, 1]`` when ``max > min``.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return scores

    lo = float(scores.min())
    hi = float(scores.max())
    span = hi - lo
    if span < eps:
        return np.full_like(scores, 0.5)

    return (scores - lo) / span


def blend_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    weight_a: float,
    normalize: bool = True,
) -> np.ndarray:
    """Blend two per-candidate score vectors for the same example.

    ``output = weight_a * A + (1 - weight_a) * B`` after optional min-max
    normalization of each vector to ``[0, 1]``. Tune ``weight_a`` on a
    validation split (grid search) to improve ``acc@k`` / MRR.

    Args:
        scores_a:   Scores from method A (e.g. HCCS), shape ``(n_chunks,)``.
        scores_b:   Scores from method B (e.g. cosine), same shape as ``scores_a``.
        weight_a:   Weight on ``scores_a``; ``1 - weight_a`` on ``scores_b``.
        normalize:  If True, min-max each vector before blending.

    Returns:
        Blended scores, same shape as inputs.

    Raises:
        ValueError: If shapes differ or ``weight_a`` is not finite.
    """
    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have the same shape")
    if not np.isfinite(weight_a):
        raise ValueError("weight_a must be finite")

    wa = float(weight_a)
    wb = 1.0 - wa
    if normalize:
        a = minmax_scores_to_unit(a)
        b = minmax_scores_to_unit(b)

    return wa * a + wb * b


def cosine_scores(
    query_emb: np.ndarray,
    chunk_embs: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity scores in embedding space."""
    if chunk_embs.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    chunk_norms = chunk_embs / (
        np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8
    )
    return chunk_norms @ query_norm


def cosine_select(
    query_emb: np.ndarray,
    chunk_embs: np.ndarray,
    top_k: int = 5,
) -> List[int]:
    """Select context chunks by cosine similarity in embedding space.

    Args:
        query_emb:  Shape ``(hidden_size,)`` — query embedding from CodeBERT.
        chunk_embs: Shape ``(n_chunks, hidden_size)`` — chunk embeddings.
        top_k:      Number of top-scoring chunks to return.

    Returns:
        List of indices sorted by descending cosine similarity.
    """
    return _select_top_k_from_scores(
        cosine_scores(query_emb, chunk_embs),
        top_k=top_k,
    )


def no_context_select() -> List[int]:
    """Return an empty selection (no context baseline).

    Returns:
        Empty list.
    """
    return []


def full_context_select(n_chunks: int) -> List[int]:
    """Select all available context chunks.

    Args:
        n_chunks: Total number of context chunks.

    Returns:
        List of all indices ``[0, 1, ..., n_chunks - 1]``.
    """
    return list(range(n_chunks))


def gold_only_select(gold_index: int) -> List[int]:
    """Oracle baseline: always select the gold snippet.

    Args:
        gold_index: The ``gold_snippet_index`` from the dataset example.

    Returns:
        Single-element list containing the gold index.
    """
    return [gold_index]
