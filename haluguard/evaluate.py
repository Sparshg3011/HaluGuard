"""
evaluate.py — Metrics computation for RepoBench next-line prediction.

All metric functions are pure Python (no ML imports required).  This makes
them fast to run and easy to test independently.

The evaluation runners (which call the full pipeline on RepoBench) are
implemented in notebooks/03_evaluation.ipynb.

Metrics defined here:
    - Exact Match (EM):    1.0 if predicted == ground truth, else 0.0
    - Edit Similarity (ES): 1 - normalised edit distance (via SequenceMatcher)
    - CodeBLEU:            Structural code similarity (via codebleu library)
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def exact_match(predicted: str, ground_truth: str) -> float:
    """Check if the predicted line exactly matches the ground truth.

    Comparison is performed on stripped strings to ignore leading/trailing
    whitespace.

    Args:
        predicted:    The model's predicted next line.
        ground_truth: The actual next line from the dataset.

    Returns:
        ``1.0`` if the stripped strings are identical, ``0.0`` otherwise.
    """
    return 1.0 if predicted.strip() == ground_truth.strip() else 0.0


def edit_similarity(predicted: str, ground_truth: str) -> float:
    """Compute character-level edit similarity between two strings.

    Uses ``difflib.SequenceMatcher`` to compute the ratio of matching
    characters.  A ratio of 1.0 means identical strings; 0.0 means
    completely different.

    Args:
        predicted:    The model's predicted next line.
        ground_truth: The actual next line from the dataset.

    Returns:
        Similarity ratio in ``[0.0, 1.0]``.
    """
    return SequenceMatcher(
        None, predicted.strip(), ground_truth.strip()
    ).ratio()


def compute_codebleu(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute CodeBLEU score over a batch of predictions.

    Wraps the ``codebleu`` library.  Falls back to 0.0 if the library is
    not installed or if computation fails.

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

def compute_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute all three metrics over a batch of predictions.

    Args:
        predictions: List of predicted next-line strings.
        references:  List of ground-truth next-line strings (same length).

    Returns:
        Dict with keys ``"em"`` (Exact Match), ``"es"`` (Edit Similarity),
        and ``"codebleu"`` (CodeBLEU).
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
    """Compute a summary metrics table across multiple methods.

    Args:
        results_by_method: Dict mapping method name → list of
                           ``(predicted, ground_truth)`` tuples.

    Returns:
        List of dicts, one per method, with keys:
        ``method``, ``em``, ``es``, ``codebleu``.
        Sorted by descending EM.
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
