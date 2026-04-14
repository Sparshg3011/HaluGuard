"""
evaluate.py — Metrics computation and benchmark loading.

Benchmark: RepoBench v1.1 (``tianyang/repobench_python_v1.1``, split ``cross_file_first``)

All metric functions are pure Python (no ML imports required).

Metrics:
    - Exact Match (EM):     1.0 if predicted == ground_truth, else 0.0
    - Edit Similarity (ES): 1 - normalised edit distance (via SequenceMatcher)
    - CodeBLEU:             Structural code similarity (via codebleu library)

Retrieval metrics (for benchmark comparison):
    - Recall@K / Accuracy@K
    - MRR (Mean Reciprocal Rank)
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core generation-quality metrics
# ---------------------------------------------------------------------------

def exact_match(predicted: str, ground_truth: str) -> float:
    """Check if the predicted line exactly matches the ground truth (stripped).

    Args:
        predicted:    The model's predicted next line.
        ground_truth: The actual next line from the dataset.

    Returns:
        ``1.0`` if equal, ``0.0`` otherwise.
    """
    return 1.0 if predicted.strip() == ground_truth.strip() else 0.0


def edit_similarity(predicted: str, ground_truth: str) -> float:
    """Character-level edit similarity using ``difflib.SequenceMatcher``.

    Args:
        predicted:    The model's predicted next line.
        ground_truth: The actual next line from the dataset.

    Returns:
        Similarity ratio in ``[0.0, 1.0]``.
    """
    return SequenceMatcher(None, predicted.strip(), ground_truth.strip()).ratio()


def compute_codebleu(predictions: List[str], references: List[str]) -> float:
    """Compute CodeBLEU score over a batch of predictions.

    Falls back to 0.0 if the ``codebleu`` library is not installed.

    Args:
        predictions: List of predicted code strings.
        references:  List of ground-truth code strings (same length).

    Returns:
        CodeBLEU score in ``[0.0, 1.0]``.
    """
    try:
        from codebleu import calc_codebleu

        result = calc_codebleu(
            references=[[ref] for ref in references],
            predictions=predictions,
            lang="python",
        )
        return float(result["codebleu"])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute EM, ES, and CodeBLEU over a batch of predictions.

    Args:
        predictions: List of predicted next-line strings.
        references:  List of ground-truth next-line strings (same length).

    Returns:
        Dict with keys ``"em"``, ``"es"``, ``"codebleu"``.
    """
    if not predictions:
        return {"em": 0.0, "es": 0.0, "codebleu": 0.0}

    em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    es_scores = [edit_similarity(p, r) for p, r in zip(predictions, references)]

    return {
        "em": sum(em_scores) / len(em_scores),
        "es": sum(es_scores) / len(es_scores),
        "codebleu": compute_codebleu(predictions, references),
    }


def compute_metrics_table(
    results_by_method: Dict[str, List[Tuple[str, str]]],
) -> List[Dict[str, Any]]:
    """Build a summary metrics table across multiple methods, sorted by EM.

    Args:
        results_by_method: Method name → list of ``(predicted, ground_truth)``
                           tuples.

    Returns:
        List of dicts with keys ``"method"``, ``"em"``, ``"es"``, ``"codebleu"``,
        sorted by descending EM.
    """
    rows: List[Dict[str, Any]] = []
    for method, pairs in results_by_method.items():
        if not pairs:
            rows.append({"method": method, "em": 0.0, "es": 0.0, "codebleu": 0.0})
            continue
        preds = [p for p, _ in pairs]
        refs = [r for _, r in pairs]
        metrics = compute_metrics(preds, refs)
        rows.append({"method": method, **metrics})
    return sorted(rows, key=lambda r: r["em"], reverse=True)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(gold_rank: int, k: int) -> float:
    """Return 1.0 if the gold chunk is ranked within the top-k, else 0.0.

    Args:
        gold_rank: 1-based position of the gold chunk in the ranked list.
        k:         Cut-off position.

    Returns:
        ``1.0`` if ``gold_rank <= k``, else ``0.0``.
    """
    return 1.0 if gold_rank <= k else 0.0


def mean_reciprocal_rank(gold_ranks: List[int]) -> float:
    """Compute MRR over a list of 1-based gold chunk ranks.

    Args:
        gold_ranks: Per-example 1-based ranks of the gold chunk.

    Returns:
        Mean reciprocal rank in ``(0.0, 1.0]``.
    """
    if not gold_ranks:
        return 0.0
    return sum(1.0 / r for r in gold_ranks) / len(gold_ranks)


def compute_retrieval_summary(
    gold_ranks: List[int],
    top_ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute MRR and Recall@K from a list of gold chunk ranks.

    Args:
        gold_ranks: Per-example 1-based ranks of the gold chunk.
        top_ks:     K values for recall computation.  Default ``[1, 3, 5]``.

    Returns:
        Dict with keys ``"mrr"``, ``"recall@1"``, ``"recall@3"``, ``"recall@5"``.
    """
    if top_ks is None:
        top_ks = [1, 3, 5]

    result: Dict[str, float] = {"mrr": mean_reciprocal_rank(gold_ranks)}
    for k in top_ks:
        hits = sum(1 for r in gold_ranks if r <= k)
        result[f"recall@{k}"] = hits / max(len(gold_ranks), 1)
    return result


