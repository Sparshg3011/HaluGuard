"""
evaluate.py — Metrics computation and benchmark loading.

Benchmark: RepoBench v1.1 (``tianyang/repobench_python_v1.1``, split ``cross_file_first``)

All metric functions are pure Python (no ML imports required).

Metrics:
    - Exact Match (EM):       1.0 if predicted == ground_truth, else 0.0
    - Edit Similarity (ES):   1 - normalised edit distance (via SequenceMatcher)
    - CodeBLEU:               Structural code similarity (via codebleu library)
    - ChrF:                   Character n-gram F-score (via sacrebleu); shown to
                              correlate better with human judgment than BLEU for
                              code ("Out of the BLEU", Evtikhiev et al. 2023)
    - Identifier F1:          Token-level F1 over Python identifiers extracted from
                              prediction vs. reference; from CrossCodeEval (NeurIPS
                              2023) — directly tests whether API/variable names match
    - Parse success:          1.0 if cropped_code + predicted_line parses as a
                              valid Python AST (or the line alone parses as a
                              statement), else 0.0.  Cheap sanity check for
                              syntactic hallucinations.

Retrieval metrics (for benchmark comparison):
    - Recall@K / Accuracy@K
    - MRR (Mean Reciprocal Rank)
    - NDCG@K
"""

from __future__ import annotations

import ast
import keyword
import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple


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
    # Drop pairs where either string is empty — they make AST parsers crash.
    pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not pairs:
        return 0.0
    preds_clean = [p for p, _ in pairs]
    refs_clean  = [r for _, r in pairs]

    try:
        from codebleu import calc_codebleu
    except ImportError:
        return 0.0

    # Try newer k4black API (list-of-lists references) first, then older flat API.
    for refs_fmt in ([[r] for r in refs_clean], refs_clean):
        try:
            result = calc_codebleu(
                references=refs_fmt,
                predictions=preds_clean,
                lang="python",
            )
            # Key is "codebleu" in k4black, "CodeBLEU" in older Microsoft version.
            score = result.get("codebleu") or result.get("CodeBLEU") or 0.0
            return float(score)
        except TypeError:
            continue  # wrong references format — try the other one
        except Exception as exc:
            print(f"[evaluate] CodeBLEU error ({type(exc).__name__}: {exc}) — returning 0.0")
            return 0.0
    return 0.0


def chrf(predicted: str, ground_truth: str, char_order: int = 6, beta: float = 1.0) -> float:
    """Compute ChrF score between a predicted and reference line.

    ChrF (character n-gram F-score) correlates better with human judgment than
    BLEU for code generation tasks (Evtikhiev et al., "Out of the BLEU", 2023).
    Uses ``sacrebleu`` if available, falls back to a lightweight built-in
    implementation so the metric is always available.

    Args:
        predicted:    The model's predicted next line.
        ground_truth: The actual next line from the dataset.
        char_order:   Maximum character n-gram order.  Default 6 (sacrebleu default).
        beta:         F-score beta weight (1.0 = equal precision/recall).

    Returns:
        ChrF score in ``[0.0, 1.0]``.
    """
    pred = predicted.strip()
    ref = ground_truth.strip()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    try:
        from sacrebleu.metrics import CHRF
        metric = CHRF(char_order=char_order, beta=beta)
        return metric.sentence_score(pred, [ref]).score / 100.0
    except ImportError:
        pass

    # Built-in fallback: character n-gram precision + recall averaged over orders.
    def _ngrams(s: str, n: int) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for i in range(len(s) - n + 1):
            ng = s[i:i + n]
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    precisions, recalls = [], []
    for n in range(1, char_order + 1):
        pred_ng = _ngrams(pred, n)
        ref_ng = _ngrams(ref, n)
        if not pred_ng or not ref_ng:
            continue
        overlap = sum(min(pred_ng.get(k, 0), ref_ng.get(k, 0)) for k in ref_ng)
        precisions.append(overlap / sum(pred_ng.values()))
        recalls.append(overlap / sum(ref_ng.values()))

    if not precisions:
        return 0.0

    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def _extract_identifiers(code: str) -> List[str]:
    """Extract Python identifiers from a code string, excluding keywords.

    Tries AST parsing first for accuracy; falls back to regex tokenisation
    if the line does not parse (e.g. it is an incomplete expression).

    Args:
        code: A Python code string (typically a single line).

    Returns:
        Ordered list of identifier strings (may contain duplicates).
    """
    try:
        tree = ast.parse(code.strip(), mode="eval")
        ids = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        # Also capture attribute names (e.g. "append" in list.append)
        attrs = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
        return ids + attrs
    except SyntaxError:
        pass

    # Regex fallback: grab all \w+ tokens that are not Python keywords or numbers
    tokens = re.findall(r"\b([A-Za-z_]\w*)\b", code)
    return [t for t in tokens if not keyword.iskeyword(t)]


def identifier_f1(predicted: str, ground_truth: str) -> float:
    """Token-level F1 over Python identifiers in predicted vs. reference line.

    Based on the CrossCodeEval (NeurIPS 2023) evaluation protocol, which uses
    identifier F1 to directly measure whether the model produces the correct
    API and variable names — the most failure-prone aspect of cross-file
    code completion.

    Args:
        predicted:    The model's predicted next line.
        ground_truth: The actual next line from the dataset.

    Returns:
        F1 score in ``[0.0, 1.0]``.  Returns ``1.0`` when both lines have no
        identifiers (e.g. pure literals / punctuation).
    """
    pred_ids = _extract_identifiers(predicted)
    ref_ids  = _extract_identifiers(ground_truth)

    if not pred_ids and not ref_ids:
        return 1.0
    if not pred_ids or not ref_ids:
        return 0.0

    # Multiset intersection
    pred_counts: Dict[str, int] = {}
    for t in pred_ids:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    ref_counts: Dict[str, int] = {}
    for t in ref_ids:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = sum(min(pred_counts.get(t, 0), ref_counts[t]) for t in ref_counts)

    precision = overlap / len(pred_ids)
    recall    = overlap / len(ref_ids)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_success(predicted: str, prefix: Optional[str] = None) -> float:
    """Return 1.0 if the predicted line parses as valid Python, else 0.0.

    When a ``prefix`` (e.g. the cropped code written so far) is supplied, the
    prefix + predicted line is parsed together so that dangling indentation or
    unclosed blocks in the prefix do not cause false negatives.  If that fails,
    we fall back to parsing the predicted line on its own.  This is a cheap
    syntactic-hallucination check — complementary to semantic metrics like
    identifier F1 and CodeBLEU.

    Args:
        predicted: The model's predicted next line.
        prefix:    Optional code written so far (e.g. ``ex["cropped_code"]``).

    Returns:
        ``1.0`` if any parse attempt succeeds, else ``0.0``.  Empty
        predictions return ``0.0``.
    """
    line = predicted.strip()
    if not line:
        return 0.0

    if prefix is not None:
        try:
            ast.parse(prefix + "\n" + predicted)
            return 1.0
        except SyntaxError:
            pass

    try:
        ast.parse(line)
        return 1.0
    except SyntaxError:
        # Some valid Python lines (e.g. ``return x``) only parse inside a
        # function body — wrap once and retry.
        try:
            ast.parse("def _wrap():\n    " + line)
            return 1.0
        except SyntaxError:
            return 0.0


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: List[str],
    references: List[str],
    prefixes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute EM, ES, CodeBLEU, ChrF, Identifier F1, and parse-success.

    Args:
        predictions: List of predicted next-line strings.
        references:  List of ground-truth next-line strings (same length).
        prefixes:    Optional cropped-code prefixes (same length).  Supplied
                     to :func:`parse_success` so that indentation-sensitive
                     continuation lines can still parse.  If omitted, the
                     predicted line is parsed standalone.

    Returns:
        Dict with keys ``"em"``, ``"es"``, ``"codebleu"``, ``"chrf"``,
        ``"id_f1"``, ``"parse_ok"``.
    """
    empty = {"em": 0.0, "es": 0.0, "codebleu": 0.0,
             "chrf": 0.0, "id_f1": 0.0, "parse_ok": 0.0}
    if not predictions:
        return empty

    em_scores   = [exact_match(p, r)      for p, r in zip(predictions, references)]
    es_scores   = [edit_similarity(p, r)  for p, r in zip(predictions, references)]
    chrf_scores = [chrf(p, r)             for p, r in zip(predictions, references)]
    id_f1s      = [identifier_f1(p, r)    for p, r in zip(predictions, references)]

    if prefixes is not None:
        parse_scores = [parse_success(p, prefix=pre)
                        for p, pre in zip(predictions, prefixes)]
    else:
        parse_scores = [parse_success(p) for p in predictions]

    return {
        "em":       sum(em_scores)     / len(em_scores),
        "es":       sum(es_scores)     / len(es_scores),
        "codebleu": compute_codebleu(predictions, references),
        "chrf":     sum(chrf_scores)   / len(chrf_scores),
        "id_f1":    sum(id_f1s)        / len(id_f1s),
        "parse_ok": sum(parse_scores)  / len(parse_scores),
    }


def compute_metrics_table(
    results_by_method: Dict[str, List[Tuple[str, str]]],
    prefixes_by_method: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Build a summary metrics table across multiple methods, sorted by EM.

    Args:
        results_by_method:  Method name → list of ``(predicted, ground_truth)``
                            tuples.
        prefixes_by_method: Optional method name → list of cropped-code
                            prefixes (same length as results) used for
                            :func:`parse_success`.  If omitted, predictions
                            are parsed standalone.

    Returns:
        List of dicts with keys ``"method"``, ``"em"``, ``"es"``,
        ``"codebleu"``, ``"chrf"``, ``"id_f1"``, ``"parse_ok"``, sorted by
        descending EM.
    """
    empty = {"em": 0.0, "es": 0.0, "codebleu": 0.0,
             "chrf": 0.0, "id_f1": 0.0, "parse_ok": 0.0}
    rows: List[Dict[str, Any]] = []
    for method, pairs in results_by_method.items():
        if not pairs:
            rows.append({"method": method, **empty})
            continue
        preds = [p for p, _ in pairs]
        refs  = [r for _, r in pairs]
        prefixes = (prefixes_by_method or {}).get(method)
        metrics = compute_metrics(preds, refs, prefixes=prefixes)
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


def ndcg_at_k(gold_ranks: List[int], k: int) -> float:
    """Compute NDCG@k over a list of 1-based gold chunk ranks.

    For single-gold retrieval (one relevant item per query), DCG@k =
    1/log2(rank+1) when the gold item is in top-k, and the ideal DCG is
    always 1/log2(2) = 1.0 (gold ranked first).

    Args:
        gold_ranks: Per-example 1-based ranks of the gold chunk.
        k:          Cut-off position.

    Returns:
        NDCG@k averaged over all examples, in ``[0.0, 1.0]``.
    """
    if not gold_ranks:
        return 0.0
    idcg = 1.0 / math.log2(2)  # ideal: gold always at rank 1
    scores = []
    for rank in gold_ranks:
        dcg = (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
        scores.append(dcg / idcg)
    return sum(scores) / len(scores)


def compute_retrieval_summary(
    gold_ranks: List[int],
    top_ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute MRR, NDCG@k, and Recall@K from a list of gold chunk ranks.

    Args:
        gold_ranks: Per-example 1-based ranks of the gold chunk.
        top_ks:     K values for recall/NDCG computation.  Default ``[1, 3, 5]``.

    Returns:
        Dict with keys ``"mrr"``, ``"recall@k"``, ``"ndcg@k"`` for each k.
    """
    if top_ks is None:
        top_ks = [1, 3, 5]

    result: Dict[str, float] = {"mrr": mean_reciprocal_rank(gold_ranks)}
    for k in top_ks:
        hits = sum(1 for r in gold_ranks if r <= k)
        result[f"recall@{k}"] = hits / max(len(gold_ranks), 1)
        result[f"ndcg@{k}"]   = ndcg_at_k(gold_ranks, k)
    return result
