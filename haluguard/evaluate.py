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
import warnings
from typing import Any, Dict, List, Optional, Tuple


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


def _wrap_for_ast(snippet: str) -> str:
    """Wrap a (possibly single-line) snippet in a function body so tree-sitter
    can parse it. CodeBLEU's syntax / dataflow components silently score 0
    when handed a fragment that isn't a complete AST node, which is exactly
    the situation for next-line completion. Indenting under a stub ``def``
    makes the snippet a parseable body without changing any tokens that
    matter for ngram/weighted-ngram match.
    """
    s = snippet.rstrip("\n")
    if not s.strip():
        return s
    indented = "\n".join("    " + line for line in s.splitlines())
    return "def _hg_wrap():\n" + indented


def _codebleu_one(pred: str, ref: str, calc_fn) -> Tuple[Optional[float], Optional[str]]:
    """Score a single (pred, ref) pair.

    Returns ``(score, error_str)``. ``error_str`` is ``None`` on success; otherwise
    it describes the exception so the caller can surface it.
    """
    p = _wrap_for_ast(pred)
    r = _wrap_for_ast(ref)
    last_err: Optional[str] = None
    # k4black expects references as list-of-lists; older Microsoft port uses a flat list.
    for refs_fmt in ([[r]], [r]):
        try:
            result = calc_fn(references=refs_fmt, predictions=[p], lang="python")
            score = result.get("codebleu")
            if score is None:
                score = result.get("CodeBLEU")
            if score is None:
                last_err = f"result dict missing 'codebleu' key: {list(result.keys())}"
                continue
            return float(score), None
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            # Only retry the other refs format on TypeError (wrong signature).
            if not isinstance(exc, TypeError):
                break
    return None, last_err


def _ngram_bleu_fallback(predictions: List[str], references: List[str]) -> Optional[float]:
    """Corpus-BLEU fallback that works without tree-sitter.

    Used when codebleu's AST/dataflow components can't load (e.g. tree-sitter
    version mismatch). Whitespace-tokenised BLEU with smoothing — not the same
    as real CodeBLEU but a meaningful non-zero signal instead of 0.0.
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore
    except ImportError:
        return None
    refs = [[r.split()] for r in references]
    hyps = [p.split() for p in predictions]
    sf = SmoothingFunction().method1
    try:
        return float(corpus_bleu(refs, hyps, smoothing_function=sf))
    except Exception:
        return None


def compute_codebleu(predictions: List[str], references: List[str]) -> float:
    """Compute CodeBLEU averaged over a batch of predictions.

    Scores each (pred, ref) pair *individually* so a single AST-parse failure
    cannot zero out the whole batch. On the first failure, prints the actual
    exception so the underlying cause (usually a tree-sitter version mismatch
    on Colab) is visible. If every codebleu call fails, falls back to an
    ngram-BLEU proxy via nltk so the column isn't uniformly 0.

    Args:
        predictions: List of predicted code strings.
        references:  List of ground-truth code strings (same length).

    Returns:
        Mean score in ``[0.0, 1.0]``.
    """
    pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not pairs:
        return 0.0

    try:
        from codebleu import calc_codebleu
    except ImportError:
        warnings.warn(
            "CodeBLEU is unavailable because the optional 'codebleu' package is "
            "not installed. Install it with `pip install -e \".[dev,codebleu]\"` "
            "or `pip install codebleu`.",
            RuntimeWarning,
            stacklevel=2,
        )
        fb = _ngram_bleu_fallback([p for p, _ in pairs], [r for _, r in pairs])
        return fb if fb is not None else 0.0

    scores: List[float] = []
    failures = 0
    first_err: Optional[str] = None
    for pred, ref in pairs:
        s, err = _codebleu_one(pred, ref, calc_codebleu)
        if s is None:
            failures += 1
            if first_err is None and err is not None:
                first_err = err
        else:
            scores.append(s)

    if not scores:
        reason = first_err or "unknown reason"
        print(f"[evaluate] CodeBLEU: all {failures} pairs failed — first error: {reason}")
        fb = _ngram_bleu_fallback([p for p, _ in pairs], [r for _, r in pairs])
        if fb is not None:
            print(f"[evaluate] CodeBLEU: falling back to nltk corpus-BLEU proxy = {fb:.4f}")
            return fb
        print("[evaluate] CodeBLEU: nltk not available for fallback — returning 0.0")
        return 0.0
    if failures:
        print(f"[evaluate] CodeBLEU: {failures}/{len(pairs)} pairs failed to score "
              f"(first error: {first_err}) — averaging {len(scores)} successful pairs")
    return sum(scores) / len(scores)


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

