"""
evaluate.py — Metrics computation and benchmark loading.

Benchmark: RepoBench v1.1 (``tianyang/repobench_python_v1.1``, split ``cross_file_first``)

All metric functions are pure Python (no ML imports required).

Metrics:
    - Exact Match (EM) and whitespace-normalized EM
    - Edit Similarity (ES): 1 - normalised edit distance (via SequenceMatcher)
    - CodeBLEU:             Structural code similarity (via codebleu library)
    - BLEU-4, chrF++, ROUGE-L
    - Identifier precision/recall/F1/EM
    - Optional CodeBERT semantic similarity
    - Syntax validity, non-empty rate, and prediction length

Retrieval metrics (for benchmark comparison):
    - Recall@K / Accuracy@K
    - MRR (Mean Reciprocal Rank)
"""

from __future__ import annotations

import ast
import builtins as _builtins
import io
import keyword
import re
import tokenize
import warnings
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


MetricValue = Optional[float]


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


def _normalise_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space."""
    return " ".join(text.strip().split())


def exact_match_normalized(predicted: str, ground_truth: str) -> float:
    """Exact match after whitespace normalization.

    This keeps EM strict on code tokens while avoiding accidental failures from
    repeated spaces, tabs, or trailing newlines in model output.
    """
    return 1.0 if _normalise_ws(predicted) == _normalise_ws(ground_truth) else 0.0


def edit_similarity(predicted: str, ground_truth: str) -> float:
    """Character-level edit similarity using ``difflib.SequenceMatcher``.

    Args:
        predicted:    The model's predicted next line.
        ground_truth: The actual next line from the dataset.

    Returns:
        Similarity ratio in ``[0.0, 1.0]``.
    """
    return SequenceMatcher(None, predicted.strip(), ground_truth.strip()).ratio()


def syntax_valid(predicted: str, ground_truth: str = "") -> float:
    """Return 1.0 if a generated Python fragment parses, else 0.0.

    Next-line completions are often fragments such as ``return x`` that are not
    valid as a module on their own. The metric therefore tries both the raw
    fragment and the same fragment indented under a stub function.
    """
    _ = ground_truth
    if not predicted.strip():
        return 0.0

    candidates = [predicted, _wrap_for_ast(predicted)]
    for candidate in candidates:
        try:
            ast.parse(candidate)
            return 1.0
        except SyntaxError:
            continue
    return 0.0


def non_empty(predicted: str, ground_truth: str = "") -> float:
    """Return 1.0 if the prediction contains any non-whitespace character."""
    _ = ground_truth
    return 1.0 if predicted.strip() else 0.0


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


def compute_codebleu_optional(
    predictions: List[str],
    references: List[str],
) -> Optional[float]:
    """Compute real CodeBLEU, returning ``None`` when unavailable.

    Unlike :func:`compute_codebleu`, this helper intentionally does not fall
    back to n-gram BLEU. Evaluation tables can then render ``NA`` instead of a
    misleading zero when CodeBLEU cannot load tree-sitter/parser dependencies.
    """
    pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not pairs:
        return 0.0

    try:
        from codebleu import calc_codebleu
    except ImportError:
        warnings.warn(
            "CodeBLEU is unavailable because the optional 'codebleu' package is "
            "not installed. Reporting CodeBLEU as NA in the metrics table.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    scores: List[float] = []
    failures = 0
    first_err: Optional[str] = None
    for pred, ref in pairs:
        score, err = _codebleu_one(pred, ref, calc_codebleu)
        if score is None:
            failures += 1
            if first_err is None:
                first_err = err
        else:
            scores.append(score)

    if not scores:
        warnings.warn(
            "CodeBLEU failed for every pair; reporting CodeBLEU as NA. "
            f"First error: {first_err or 'unknown'}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if failures:
        warnings.warn(
            f"CodeBLEU failed for {failures}/{len(pairs)} pairs. "
            f"First error: {first_err or 'unknown'}",
            RuntimeWarning,
            stacklevel=2,
        )
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Identifier-set metrics (CrossCodeEval native)
# ---------------------------------------------------------------------------

_PY_BUILTINS: Set[str] = set(dir(_builtins))


def identifier_tokens(src: str) -> Set[str]:
    """Return the set of identifier names in ``src``.

    Uses the stdlib :mod:`tokenize` module and keeps only ``NAME`` tokens that
    are not Python keywords or builtins. Returns an empty set if tokenisation
    fails (common for one-liner fragments that are not syntactically valid on
    their own — the fallback is a conservative whitespace-word split).

    Args:
        src: Source code fragment.

    Returns:
        Set of identifier strings.
    """
    if not src or not src.strip():
        return set()
    names: Set[str] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.NAME:
                name = tok.string
                if name in _PY_BUILTINS or keyword.iskeyword(name):
                    continue
                names.add(name)
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        # Fallback: crude whitespace split keeping identifier-shaped words.
        for word in src.replace(".", " ").replace("(", " ").replace(")", " ").split():
            stripped = word.strip("_")
            if stripped and stripped[0].isalpha() and stripped.isidentifier():
                if stripped in _PY_BUILTINS or keyword.iskeyword(stripped):
                    continue
                names.add(word if word.isidentifier() else stripped)
    return names


def identifier_em(predicted: str, ground_truth: str) -> float:
    """Exact match over identifier sets.

    Returns ``1.0`` iff the two sets of identifier names are equal.
    Empty-vs-empty counts as a match.
    """
    return 1.0 if identifier_tokens(predicted) == identifier_tokens(ground_truth) else 0.0


def identifier_precision(predicted: str, ground_truth: str) -> float:
    """Precision over identifier sets.

    Empty-vs-empty counts as perfect. Empty prediction against a non-empty
    reference counts as zero.
    """
    pred = identifier_tokens(predicted)
    ref = identifier_tokens(ground_truth)
    if not pred and not ref:
        return 1.0
    if not pred:
        return 0.0
    return len(pred & ref) / len(pred)


def identifier_recall(predicted: str, ground_truth: str) -> float:
    """Recall over identifier sets."""
    pred = identifier_tokens(predicted)
    ref = identifier_tokens(ground_truth)
    if not pred and not ref:
        return 1.0
    if not ref:
        return 0.0
    return len(pred & ref) / len(ref)


def identifier_f1(predicted: str, ground_truth: str) -> float:
    """F1 over identifier sets.

    F1 = 2 * precision * recall / (precision + recall). If both sets are empty
    the score is ``1.0`` (vacuous match). If exactly one is empty, ``0.0``.
    """
    pred = identifier_tokens(predicted)
    ref = identifier_tokens(ground_truth)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(ref)
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# n-gram / surface metrics (sacrebleu, rouge_score)
# ---------------------------------------------------------------------------

def bleu4(predicted: str, ground_truth: str) -> float:
    """Sentence-BLEU-4 with smoothing.

    Uses ``sacrebleu.sentence_bleu`` if available and falls back to nltk's
    ``sentence_bleu`` with smoothing method 1 if only nltk is installed.
    Returns ``0.0`` if neither library is available (silently — the metric is
    simply skipped from aggregation).
    """
    if not predicted.strip() and not ground_truth.strip():
        return 1.0
    try:
        import sacrebleu  # type: ignore

        return float(sacrebleu.sentence_bleu(predicted, [ground_truth]).score) / 100.0
    except ImportError:
        pass
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore

        sf = SmoothingFunction().method1
        return float(
            sentence_bleu(
                [ground_truth.split()],
                predicted.split(),
                smoothing_function=sf,
            )
        )
    except ImportError:
        return 0.0


def chrf_plus(predicted: str, ground_truth: str) -> float:
    """Sentence-level chrF++ score in [0, 1].

    chrF++ is a character-n-gram F-score augmented with word n-grams. It is
    language-agnostic and robust to short one-liner outputs where BLEU is
    noisy. Returns ``0.0`` if ``sacrebleu`` is not installed.
    """
    if not predicted.strip() and not ground_truth.strip():
        return 1.0
    try:
        import sacrebleu  # type: ignore

        return float(
            sacrebleu.sentence_chrf(predicted, [ground_truth], word_order=2).score
        ) / 100.0
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def rouge_l(predicted: str, ground_truth: str) -> float:
    """ROUGE-L F1 score (longest-common-subsequence).

    Uses the ``rouge_score`` library. Returns ``0.0`` if not installed.
    """
    if not predicted.strip() and not ground_truth.strip():
        return 1.0
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        return 0.0
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        score = scorer.score(ground_truth, predicted)
        return float(score["rougeL"].fmeasure)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CodeBERTScore — semantic F1 via mean-pooled embeddings
# ---------------------------------------------------------------------------

def codebert_score(
    predictions: List[str],
    references: List[str],
    encoder: Any,
    tokenizer: Any,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 256,
) -> List[float]:
    """Cosine similarity between mean-pooled CodeBERT embeddings.

    A cheap, batched semantic metric that reuses the encoder already loaded by
    the pipeline. Not identical to the CodeBERTScore paper (which uses F1 over
    per-token similarities) but is well-correlated and ~20× faster on short
    next-line outputs.

    Args:
        predictions: Predicted next-line strings.
        references:  Ground-truth next-line strings (same length).
        encoder:     Frozen HuggingFace encoder in eval mode.
        tokenizer:   Matching tokenizer.
        device:      Torch device; inferred if omitted.
        batch_size:  Encoder mini-batch size.
        max_length:  Token truncation length.

    Returns:
        List of floats in ``[-1, 1]`` (typically ``[0, 1]`` for code), one per
        pair.
    """
    if not predictions:
        return []

    try:
        import torch  # type: ignore
    except ImportError:
        return [0.0] * len(predictions)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _encode(texts: List[str]) -> "torch.Tensor":
        outs: List["torch.Tensor"] = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                hidden = encoder(**enc).last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            outs.append(pooled.detach().cpu())
        return torch.cat(outs, dim=0)

    pred_emb = _encode(predictions)
    ref_emb = _encode(references)
    scores = (pred_emb * ref_emb).sum(dim=-1)
    return [float(s) for s in scores.tolist()]


# ---------------------------------------------------------------------------
# EFL error-type histogram
# ---------------------------------------------------------------------------

def first_error_type_hist(histories: Sequence[Any]) -> Dict[str, int]:
    """Tally first-iteration error types across an EFL ``history`` list.

    Each entry in ``histories`` should be either (a) a dict with a ``"history"``
    key containing a list whose first element has an ``"error_type"`` field, or
    (b) a list of :class:`ExecutionResult`-like objects (dicts or dataclass
    instances) exposing ``error_type``.

    Examples that passed on iteration 1 are tallied under the ``"none"`` key.

    Args:
        histories: Sequence of per-example EFL history objects.

    Returns:
        Dict mapping error-type name → count.
    """
    counts: Counter = Counter()

    def _first_error(entry: Any) -> Optional[str]:
        if isinstance(entry, dict) and "history" in entry:
            hist = entry["history"]
        else:
            hist = entry
        if not hist:
            return None
        first = hist[0]
        if isinstance(first, dict):
            return first.get("error_type")
        return getattr(first, "error_type", None)

    for entry in histories:
        err = _first_error(entry)
        counts[err or "none"] += 1

    return dict(counts)


# ---------------------------------------------------------------------------
# Metric dispatch
# ---------------------------------------------------------------------------

# The full set of supported string/semantic metrics, in report order.
ALL_METRICS: Tuple[str, ...] = (
    "em",
    "em_normalized",
    "es",
    "codebleu",
    "bleu_4",
    "chrf_plus",
    "rouge_l",
    "identifier_precision",
    "identifier_recall",
    "identifier_em",
    "identifier_f1",
    "codebert_score",
    "syntax_valid",
    "non_empty_rate",
    "avg_prediction_chars",
)


def metric_applies(name: str, benchmark: str) -> bool:
    """Decide whether a metric is reported for a given benchmark.

    All surface / identifier metrics apply to every Python-next-line benchmark.
    CodeBLEU is the RepoBench-native metric and still meaningful on
    CrossCodeEval. ``codebert_score`` is always applicable.
    """
    if name not in ALL_METRICS:
        return False
    # Currently we report all metrics on both benchmarks; the function exists
    # so future benchmarks (e.g. non-Python) can opt out cleanly.
    _ = benchmark
    return True


def compute_metrics(
    predictions: List[str],
    references: List[str],
    metrics: Optional[Sequence[str]] = None,
    encoder: Any = None,
    tokenizer: Any = None,
    device: Optional[str] = None,
) -> Dict[str, MetricValue]:
    """Compute a dispatchable set of metrics over a batch.

    Backwards-compatible default: if ``metrics`` is ``None``, computes EM, ES,
    and CodeBLEU (the historical shape). If ``metrics`` is a list, only those
    are computed. ``codebert_score`` additionally requires ``encoder`` and
    ``tokenizer``; if they are missing the key is reported as ``None`` so
    tables render it as ``NA`` rather than a fake zero.

    Args:
        predictions: Predicted next-line strings.
        references:  Ground-truth next-line strings (same length).
        metrics:     Subset of :data:`ALL_METRICS` to compute. ``None`` =
                     historical ``["em","es","codebleu"]``.
        encoder:     Frozen encoder for ``codebert_score``.
        tokenizer:   Tokenizer for ``codebert_score``.
        device:      Torch device; inferred if omitted.

    Returns:
        Dict of metric name → mean score over the batch.
    """
    if metrics is None:
        metrics = ("em", "es", "codebleu")

    out: Dict[str, MetricValue] = {}
    if not predictions:
        for m in metrics:
            out[m] = 0.0
        return out

    per_pair: Dict[str, Callable[[str, str], float]] = {
        "em": exact_match,
        "em_normalized": exact_match_normalized,
        "es": edit_similarity,
        "bleu_4": bleu4,
        "chrf_plus": chrf_plus,
        "rouge_l": rouge_l,
        "identifier_precision": identifier_precision,
        "identifier_recall": identifier_recall,
        "identifier_em": identifier_em,
        "identifier_f1": identifier_f1,
        "syntax_valid": syntax_valid,
        "non_empty_rate": non_empty,
    }

    for name in metrics:
        if name in per_pair:
            fn = per_pair[name]
            scores = [fn(p, r) for p, r in zip(predictions, references)]
            out[name] = sum(scores) / len(scores)
        elif name == "codebleu":
            out[name] = compute_codebleu_optional(predictions, references)
        elif name == "codebert_score":
            if encoder is None or tokenizer is None:
                out[name] = None
                continue
            scores = codebert_score(
                predictions, references, encoder, tokenizer, device=device
            )
            out[name] = sum(scores) / len(scores) if scores else 0.0
        elif name == "avg_prediction_chars":
            lengths = [len(p.rstrip("\n")) for p in predictions]
            out[name] = sum(lengths) / len(lengths)
        else:
            # Unknown metric name — skip rather than crash.
            continue

    return out


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
