"""
type_router.py — Pre-emptive context boosting based on code pattern analysis.

Operates in two modes:

**Pre-emptive (before generation):**
    Analyse ``cropped_code`` to predict which error types the model is likely
    to produce, then boost HCCS scores for context chunks that would prevent
    those errors.

**Post-failure (EFL retry):**
    Map the actual Python exception to a hallucination category and boost
    chunks matching that category.

The mapping is intentionally rule-based (not learned) because the relationship
between error type and remediation context is logical and deterministic.
See docs/DECISIONS.md, Decision 3 for the full rationale.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from haluguard.hccs import HallucinationType


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

ERROR_TO_CATEGORY: Dict[str, str] = {
    # RESOURCE: missing imports / packages
    "ImportError":          HallucinationType.RESOURCE.value,
    "ModuleNotFoundError":  HallucinationType.RESOURCE.value,

    # NAMING: wrong identifier, attribute, or unbound local
    "NameError":            HallucinationType.NAMING.value,
    "AttributeError":       HallucinationType.NAMING.value,
    "UnboundLocalError":    HallucinationType.NAMING.value,

    # MAPPING: wrong key, index, or argument type
    "KeyError":             HallucinationType.MAPPING.value,
    "IndexError":           HallucinationType.MAPPING.value,
    "TypeError":            HallucinationType.MAPPING.value,

    # LOGIC: wrong values, assertions, or general runtime errors
    "ValueError":           HallucinationType.LOGIC.value,
    "AssertionError":       HallucinationType.LOGIC.value,
    "RuntimeError":         HallucinationType.LOGIC.value,
    "ZeroDivisionError":    HallucinationType.LOGIC.value,
    "RecursionError":       HallucinationType.LOGIC.value,
    "StopIteration":        HallucinationType.LOGIC.value,
}


# ---------------------------------------------------------------------------
# Pre-emptive analysis of cropped_code
# ---------------------------------------------------------------------------

def predict_boost(cropped_code: str) -> Dict[str, float]:
    """Analyse ``cropped_code`` to predict which context types are most needed.

    Uses lightweight regex patterns to detect code patterns that correlate
    with specific hallucination types.  Returns a dict of additive boosts
    keyed by ``HallucinationType.value``.

    Args:
        cropped_code: The code written so far in the current file.

    Returns:
        Dict mapping hallucination type value (e.g. ``"naming"``) to a float
        boost (typically 0.0–0.15).  Types not detected get 0.0.
    """
    boosts: Dict[str, float] = {h.value: 0.0 for h in HallucinationType}

    # Method calls like obj.method() → model needs to know class definitions
    if re.search(r"\w+\.\w+\(", cropped_code):
        boosts[HallucinationType.NAMING.value] += 0.15

    # Import statements → model needs to know available modules
    if re.search(r"from\s+\S+\s+import|import\s+\S+", cropped_code):
        boosts[HallucinationType.RESOURCE.value] += 0.1

    # Type annotations → model needs to know type signatures
    if re.search(r":\s*(List|Dict|Optional|Tuple|Set|int|str|float|bool)\b", cropped_code):
        boosts[HallucinationType.MAPPING.value] += 0.1

    # Assertions or test patterns → model needs logic context
    if re.search(r"\bassert\b|assertEqual|assertTrue", cropped_code):
        boosts[HallucinationType.LOGIC.value] += 0.1

    # Function/class definitions being constructed → naming context needed
    if re.search(r"^(class |def )", cropped_code, re.MULTILINE):
        boosts[HallucinationType.NAMING.value] += 0.05

    return boosts


# ---------------------------------------------------------------------------
# Snippet classification
# ---------------------------------------------------------------------------

def classify_snippet(snippet: str, path: str) -> Optional[str]:
    """Classify a context snippet by what hallucination type it could prevent.

    Uses lightweight heuristics on the snippet content and file path to
    determine its category.

    Args:
        snippet: Code snippet text from the context chunk.
        path:    File path of the snippet's source file.

    Returns:
        A ``HallucinationType`` value string (e.g. ``"resource"``), or
        ``None`` if the snippet does not clearly match any category.
    """
    # Import-heavy snippets → RESOURCE
    import_count = len(re.findall(
        r"^(?:from\s+\S+\s+import|import\s+\S+)", snippet, re.MULTILINE
    ))
    if import_count >= 2 or "__init__" in path:
        return HallucinationType.RESOURCE.value

    # Class/function definitions → NAMING
    has_defs = bool(re.search(r"^(?:class |def )\w+", snippet, re.MULTILINE))
    if has_defs:
        return HallucinationType.NAMING.value

    # Function signatures with type annotations → MAPPING
    has_typed_sig = bool(re.search(
        r"def \w+\(.*:\s*\w+", snippet, re.MULTILINE
    ))
    if has_typed_sig:
        return HallucinationType.MAPPING.value

    # Test files → LOGIC
    if "test" in path.lower():
        return HallucinationType.LOGIC.value

    return None


# ---------------------------------------------------------------------------
# Score boosting
# ---------------------------------------------------------------------------

def boost_scores(
    scores: np.ndarray,
    contexts: List[Dict[str, str]],
    boosts: Dict[str, float],
) -> np.ndarray:
    """Apply additive boosts to HCCS scores based on snippet classification.

    For each context chunk, classifies the snippet and adds the corresponding
    boost from ``boosts``.  Scores are capped at 1.0.

    Args:
        scores:   1-D array of HCCS scores, shape ``(n_chunks,)``.
        contexts: List of context dicts with ``"snippet"`` and ``"path"`` keys.
        boosts:   Dict mapping hallucination type value → additive boost.

    Returns:
        New array of adjusted scores (same shape as ``scores``).
    """
    adjusted = scores.copy().astype(np.float64)

    for i, ctx in enumerate(contexts):
        category = classify_snippet(ctx["snippet"], ctx["path"])
        if category is not None and category in boosts:
            adjusted[i] += boosts[category]

    return np.clip(adjusted, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Post-failure boosting (used by EFL)
# ---------------------------------------------------------------------------

def error_boost(error_type: str) -> Dict[str, float]:
    """Map an actual Python exception to boost weights for context re-ranking.

    Called by the EFL after a generation attempt fails.  Returns strong boosts
    for the category matching the error, with smaller boosts for related types.

    Args:
        error_type: Python exception class name, e.g. ``"ImportError"``.

    Returns:
        Dict mapping hallucination type value → additive boost.
    """
    boosts: Dict[str, float] = {h.value: 0.0 for h in HallucinationType}

    category = ERROR_TO_CATEGORY.get(error_type)
    if category is not None:
        # Strong boost for the matching category
        boosts[category] = 0.2
    else:
        # Unknown error → mild boost for everything
        for key in boosts:
            boosts[key] = 0.05

    return boosts
