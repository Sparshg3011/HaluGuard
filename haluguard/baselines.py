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

from typing import Any, Dict, List

import numpy as np


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

    top_k = min(top_k, len(contexts))
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return sorted_indices.tolist()


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
    if chunk_embs.shape[0] == 0:
        return []

    # Normalise to unit vectors
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    chunk_norms = chunk_embs / (
        np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8
    )
    similarities = chunk_norms @ query_norm  # (n_chunks,)

    top_k = min(top_k, len(similarities))
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    return sorted_indices.tolist()


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
