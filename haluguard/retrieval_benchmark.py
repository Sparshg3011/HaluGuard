"""
retrieval_benchmark.py - Shared helpers for RepoBench-style retrieval evaluation.

This module centralises the paper-aligned retrieval protocol used by the
notebooks:

* query views:
    - ``full``  -> full ``cropped_code``
    - ``last3`` -> last three in-file lines only
* candidate buckets:
    - ``easy``  -> 5-9 candidates
    - ``hard``  -> 10+ candidates
* ranking metrics:
    - accuracy@k (equivalent to recall@k for single-gold retrieval)
    - mean reciprocal rank (MRR)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


QUERY_VIEW_FULL = "full"
QUERY_VIEW_LAST3 = "last3"

BUCKET_EASY = "easy"
BUCKET_HARD = "hard"

DEFAULT_BUCKET_TOP_KS: Dict[str, Sequence[int]] = {
    BUCKET_EASY: (1, 3),
    BUCKET_HARD: (1, 3, 5),
}


@dataclass
class RetrievalRankingResult:
    """One evaluated retrieval ranking for a single RepoBench example."""

    method: str
    example_id: int
    query_view: str
    candidate_count: int
    gold_index: int
    ranked_indices: List[int]
    gold_rank: int
    bucket: Optional[str]


def build_query_text(cropped_code: str, query_view: str) -> str:
    """Build the retrieval query text for a given query-view protocol."""
    if query_view == QUERY_VIEW_FULL:
        return cropped_code
    if query_view == QUERY_VIEW_LAST3:
        lines = cropped_code.splitlines()
        return "\n".join(lines[-3:]) if lines else ""
    raise ValueError(f"Unsupported query view: {query_view}")


def get_query_embedding_path(
    emb_dir: Path,
    backend: str,
    query_view: str,
) -> Path:
    """Return the cached query-embedding artifact path for a variant."""
    return Path(emb_dir) / f"query_embeddings__{backend}__{query_view}.pt"


def get_query_meta_path(
    emb_dir: Path,
    backend: str,
    query_view: str,
) -> Path:
    """Return the metadata path for a cached query-embedding artifact."""
    return Path(emb_dir) / f"query_embeddings__{backend}__{query_view}.meta.json"


def get_chunk_embedding_path(
    emb_dir: Path,
    backend: str,
) -> Path:
    """Return the cached chunk-embedding artifact path for a backend."""
    return Path(emb_dir) / f"chunk_embeddings__{backend}.pt"


def get_chunk_meta_path(
    emb_dir: Path,
    backend: str,
) -> Path:
    """Return the metadata path for a cached chunk-embedding artifact."""
    return Path(emb_dir) / f"chunk_embeddings__{backend}.meta.json"


def get_hccs_checkpoint_name(
    backend: str,
    query_view: str,
) -> str:
    """Return the canonical checkpoint filename for an HCCS variant."""
    return f"hccs_{backend}_{query_view}_best.pt"


def get_candidate_bucket(candidate_count: int) -> Optional[str]:
    """Bucket RepoBench retrieval examples by candidate-set size."""
    if 5 <= candidate_count <= 9:
        return BUCKET_EASY
    if candidate_count >= 10:
        return BUCKET_HARD
    return None


def rank_indices_from_scores(scores: Sequence[float]) -> List[int]:
    """Return candidate indices ranked by descending score."""
    score_array = np.asarray(scores, dtype=np.float64)
    return np.argsort(score_array)[::-1].tolist()


def compute_gold_rank(ranked_indices: Sequence[int], gold_index: int) -> int:
    """Return the 1-based position of the gold index in a ranked list."""
    return list(ranked_indices).index(int(gold_index)) + 1


def build_ranking_result(
    method: str,
    example_id: int,
    query_view: str,
    candidate_count: int,
    gold_index: int,
    ranked_indices: Sequence[int],
) -> RetrievalRankingResult:
    """Create a ``RetrievalRankingResult`` from a ranked candidate list."""
    ranked_list = list(ranked_indices)
    return RetrievalRankingResult(
        method=method,
        example_id=int(example_id),
        query_view=query_view,
        candidate_count=int(candidate_count),
        gold_index=int(gold_index),
        ranked_indices=ranked_list,
        gold_rank=compute_gold_rank(ranked_list, gold_index),
        bucket=get_candidate_bucket(int(candidate_count)),
    )


def compute_accuracy_metrics(
    gold_ranks: Sequence[int],
    top_ks: Sequence[int],
) -> Dict[str, float]:
    """Compute accuracy@k and MRR from 1-based gold ranks."""
    ranks = [int(rank) for rank in gold_ranks]
    total = max(len(ranks), 1)

    metrics: Dict[str, float] = {
        "mrr": sum(1.0 / rank for rank in ranks) / total,
    }
    for k in top_ks:
        metrics[f"acc@{int(k)}"] = sum(rank <= int(k) for rank in ranks) / total
    return metrics


def summarise_rankings(
    rankings: Sequence[RetrievalRankingResult],
    bucket_top_ks: Optional[Dict[str, Sequence[int]]] = None,
    include_mrr: bool = False,
) -> List[Dict[str, Any]]:
    """Aggregate ranking results into paper-style easy/hard rows."""
    if bucket_top_ks is None:
        bucket_top_ks = DEFAULT_BUCKET_TOP_KS

    method_names = sorted({ranking.method for ranking in rankings})
    table: List[Dict[str, Any]] = []

    for method in method_names:
        method_rankings = [ranking for ranking in rankings if ranking.method == method]
        row: Dict[str, Any] = {"method": method}

        for bucket, top_ks in bucket_top_ks.items():
            bucket_rankings = [
                ranking for ranking in method_rankings if ranking.bucket == bucket
            ]
            bucket_metrics = compute_accuracy_metrics(
                [ranking.gold_rank for ranking in bucket_rankings],
                top_ks=top_ks,
            )
            if not include_mrr:
                bucket_metrics.pop("mrr", None)
            row[f"{bucket}_count"] = len(bucket_rankings)
            row.update(
                {
                    f"{bucket}_{metric_name}": metric_value
                    for metric_name, metric_value in bucket_metrics.items()
                }
            )

        table.append(row)

    return table


def save_rankings_jsonl(
    rankings: Iterable[RetrievalRankingResult],
    path: Path,
) -> None:
    """Save per-example rankings to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ranking in rankings:
            handle.write(json.dumps(asdict(ranking)) + "\n")


def save_table_json(
    table: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    """Save an aggregated retrieval table to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(table), handle, indent=2)
