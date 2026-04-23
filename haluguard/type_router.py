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

Two interchangeable router backends are provided:

- :class:`RegexTypeRouter` — the original rule-based implementation.
- :class:`NoOpTypeRouter`  — returns all-zero boosts (disables the router).
- :class:`LearnedTypeRouter` — CodeBERT + 4-head linear classifier trained on
  EFL error traces. Use :func:`haluguard.training.train_learned_router`.

All three satisfy :class:`TypeRouterBase` and are drop-in swappable.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn

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


def _zero_boosts() -> Dict[str, float]:
    """Return a dict of all-zero boosts, one per HallucinationType."""
    return {h.value: 0.0 for h in HallucinationType}


# ---------------------------------------------------------------------------
# Pre-emptive analysis of cropped_code (regex version)
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
    boosts = _zero_boosts()

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


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a score vector without changing empty/constant shape."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape[0] == 0:
        return arr.copy()
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def boost_scores_normalized(
    scores: np.ndarray,
    contexts: List[Dict[str, str]],
    boosts: Dict[str, float],
) -> np.ndarray:
    """Apply category boosts after normalizing raw logits/scores.

    The legacy :func:`boost_scores` assumes scores already live in ``[0, 1]``.
    New scorer architectures return raw logits, so clipping them directly can
    destroy the ranking signal. This helper keeps no-boost rankings unchanged
    and only enters normalized score space when at least one boost is active.
    """
    if not contexts or not any(float(v) != 0.0 for v in boosts.values()):
        return np.asarray(scores, dtype=np.float64).copy()
    base = normalize_scores(scores)
    return boost_scores(base, contexts, boosts)


def apply_router_boosts(
    scores: np.ndarray,
    contexts: List[Dict[str, str]],
    router: "TypeRouterBase",
    cropped_code: str,
) -> np.ndarray:
    """Apply a router to raw scores in a scorer-safe way.

    Routers may optionally expose ``context_boosts(cropped_code, contexts)`` for
    per-context additive boosts. Otherwise the generic category-level boost
    mechanism is used.
    """
    if not contexts:
        return np.asarray(scores, dtype=np.float64).copy()

    boosts = router.predict_boost(cropped_code)
    has_category_boost = any(float(v) != 0.0 for v in boosts.values())
    context_boost_fn = getattr(router, "context_boosts", None)
    context_boosts: Optional[np.ndarray] = None
    if callable(context_boost_fn):
        context_boosts = np.asarray(
            context_boost_fn(cropped_code, contexts),
            dtype=np.float64,
        )
        if context_boosts.shape != np.asarray(scores).shape:
            raise ValueError(
                "router.context_boosts must return one boost per score: "
                f"got {context_boosts.shape}, expected {np.asarray(scores).shape}"
            )

    has_context_boost = (
        context_boosts is not None and bool(np.any(np.abs(context_boosts) > 0.0))
    )
    if not has_category_boost and not has_context_boost:
        return np.asarray(scores, dtype=np.float64).copy()

    adjusted = (
        boost_scores(normalize_scores(scores), contexts, boosts)
        if has_category_boost
        else normalize_scores(scores)
    )
    if has_context_boost and context_boosts is not None:
        adjusted = np.clip(adjusted + context_boosts, 0.0, 1.0)
    return adjusted


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
    boosts = _zero_boosts()

    category = ERROR_TO_CATEGORY.get(error_type)
    if category is not None:
        # Strong boost for the matching category
        boosts[category] = 0.2
    else:
        # Unknown error → mild boost for everything
        for key in boosts:
            boosts[key] = 0.05

    return boosts


# ---------------------------------------------------------------------------
# Pluggable router protocol + implementations
# ---------------------------------------------------------------------------

class TypeRouterBase:
    """Abstract base class for pre-emptive context boosters.

    Two methods must be implemented:

    - :meth:`predict_boost` — boost dict from ``cropped_code`` (pre-generation).
    - :meth:`error_boost`   — boost dict from a Python exception class name
      (post-failure, used by EFL).

    All returned dicts MUST contain an entry for every
    :class:`HallucinationType` value with a finite float in ``[0, 1]``.
    """

    name: str = "base"

    def predict_boost(self, cropped_code: str) -> Dict[str, float]:
        raise NotImplementedError

    def error_boost(self, error_type: str) -> Dict[str, float]:
        raise NotImplementedError


class RegexTypeRouter(TypeRouterBase):
    """Rule-based router — delegates to the free functions above.

    Preserves the historical HaluGuard behaviour; no state, no training
    required. Used as the back-compat default in ``pipeline.py`` and
    ``efl.run_efl``.
    """

    name = "regex"

    def predict_boost(self, cropped_code: str) -> Dict[str, float]:
        return predict_boost(cropped_code)

    def error_boost(self, error_type: str) -> Dict[str, float]:
        return error_boost(error_type)


class NoOpTypeRouter(TypeRouterBase):
    """Returns all-zero boosts — effectively disables pre-emptive routing.

    ``error_boost`` still delegates to the rule-based mapping so EFL retries
    retain a signal even when the pre-emptive boost is off — this matches the
    ablation "no pre-emptive router but keep EFL re-ranking".
    """

    name = "noop"

    def predict_boost(self, cropped_code: str) -> Dict[str, float]:
        return _zero_boosts()

    def error_boost(self, error_type: str) -> Dict[str, float]:
        return error_boost(error_type)


def _safe_parse_module(source: str) -> Optional[ast.AST]:
    """Parse Python source, returning ``None`` for incomplete fragments."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _names_from_ast(tree: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return names


def _definition_names(snippet: str) -> Set[str]:
    tree = _safe_parse_module(snippet)
    if tree is None:
        return set(re.findall(r"^(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", snippet, re.MULTILINE))
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
    return out


def _imported_names(snippet: str) -> Set[str]:
    tree = _safe_parse_module(snippet)
    if tree is None:
        return set()
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.add(alias.asname or alias.name)
    return out


class AstSymbolRouter(TypeRouterBase):
    """AST/token based router with symbol-aware per-context boosts.

    This router keeps the same four hallucination categories as the regex
    router, but it derives the pre-generation signal from Python syntax when
    possible and boosts snippets that actually define/import names seen near
    the completion point.
    """

    name = "ast_symbol"

    def predict_boost(self, cropped_code: str) -> Dict[str, float]:
        boosts = _zero_boosts()
        tree = _safe_parse_module(cropped_code)

        if tree is None:
            # Incomplete code is common near the cursor. Fall back to the
            # conservative regex rules rather than dropping the signal.
            return predict_boost(cropped_code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                boosts[HallucinationType.RESOURCE.value] = max(
                    boosts[HallucinationType.RESOURCE.value],
                    0.10,
                )
            elif isinstance(node, ast.Attribute):
                boosts[HallucinationType.NAMING.value] = max(
                    boosts[HallucinationType.NAMING.value],
                    0.15,
                )
            elif isinstance(node, ast.Call):
                boosts[HallucinationType.NAMING.value] = max(
                    boosts[HallucinationType.NAMING.value],
                    0.10,
                )
                if node.keywords or len(node.args) >= 2:
                    boosts[HallucinationType.MAPPING.value] = max(
                        boosts[HallucinationType.MAPPING.value],
                        0.08,
                    )
            elif isinstance(node, (ast.AnnAssign, ast.arg)):
                boosts[HallucinationType.MAPPING.value] = max(
                    boosts[HallucinationType.MAPPING.value],
                    0.10,
                )
            elif isinstance(node, ast.Assert):
                boosts[HallucinationType.LOGIC.value] = max(
                    boosts[HallucinationType.LOGIC.value],
                    0.10,
                )

        return boosts

    def context_boosts(
        self,
        cropped_code: str,
        contexts: List[Dict[str, str]],
    ) -> np.ndarray:
        """Return symbol-overlap boosts, one per context chunk."""
        tree = _safe_parse_module(cropped_code)
        if tree is None:
            query_names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cropped_code))
        else:
            query_names = _names_from_ast(tree)

        boosts: List[float] = []
        for ctx in contexts:
            snippet = ctx.get("snippet", "")
            path = ctx.get("path", "")
            defs = _definition_names(snippet)
            imports = _imported_names(snippet)
            names = defs | imports
            overlap = query_names & names

            boost = 0.0
            if overlap:
                boost += min(0.12, 0.04 * len(overlap))
            if defs and re.search(r"\.\s*$|\.\w+\(", cropped_code):
                boost += 0.04
            if imports and any(name in cropped_code for name in imports):
                boost += 0.04
            if "test" in path.lower() and re.search(r"\bassert\b|assertEqual|assertTrue", cropped_code):
                boost += 0.06
            boosts.append(min(boost, 0.18))

        return np.asarray(boosts, dtype=np.float64)

    def error_boost(self, error_type: str) -> Dict[str, float]:
        return error_boost(error_type)


class LearnedTypeRouter(TypeRouterBase):
    """Learned multi-label classifier over HallucinationType categories.

    Architecture: frozen CodeBERT encoder (shared with the pipeline) →
    ``nn.Linear(768, 4)`` → sigmoid → scaled by a learned temperature so
    outputs match the [0, 0.2] magnitude the boost mechanism expects.

    Training: see :func:`haluguard.training.train_learned_router`.

    Back-pressure: ``error_boost`` still uses the rule-based mapping — the true
    Python exception is known by the time EFL calls it, so there is no
    learning problem there.

    Args:
        encoder:   Frozen HuggingFace encoder in eval mode (typically CodeBERT).
        tokenizer: Matching HuggingFace tokenizer.
        head:      ``torch.nn.Module`` with a ``forward(emb) -> logits (B, 4)``
                   interface.  If ``None``, a fresh randomly-initialised head
                   is created (useful for tests / protocol checks).
        temperature: Learned scalar (stored on the head). Lower → sharper
                     boosts. Falls back to ``0.2`` if ``head`` has no
                     ``temperature`` attribute.
        max_boost: Upper bound on any individual boost.  Defaults to 0.2 to
                   match the magnitude the pipeline expects.
        device:    Torch device string.  Inferred from CUDA availability if
                   omitted.
    """

    name = "learned"

    def __init__(
        self,
        encoder: Any = None,
        tokenizer: Any = None,
        head: Optional["LearnedRouterHead"] = None,
        max_boost: float = 0.2,
        device: Optional[str] = None,
    ) -> None:
        import torch  # local import: keep module importable without torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = encoder
        self.tokenizer = tokenizer
        self.head = head if head is not None else LearnedRouterHead()
        self.head.eval()
        self.head.to(device)
        self.max_boost = float(max_boost)
        self.device = device
        self._categories: List[str] = [h.value for h in HallucinationType]

    # ---- helpers -----------------------------------------------------

    def _embed(self, cropped_code: str) -> "torch.Tensor":
        """Embed ``cropped_code`` via the frozen encoder; return a 1-D tensor.

        If no encoder/tokenizer was supplied (e.g. in unit tests with a
        pre-computed embedding), returns a zero vector so the head still runs
        end-to-end.
        """
        import torch

        if self.encoder is None or self.tokenizer is None:
            hidden_size = getattr(self.head, "hidden_size", 768)
            return torch.zeros(hidden_size, dtype=torch.float32, device=self.device)

        # Reuse the project's embed_code helper for consistent truncation.
        from haluguard.hccs import embed_code

        emb = embed_code(
            cropped_code,
            tokenizer=self.tokenizer,
            model=self.encoder,
            device=self.device,
            truncation_side="left",
        )
        return torch.as_tensor(emb, dtype=torch.float32, device=self.device)

    # ---- API ---------------------------------------------------------

    def predict_boost(self, cropped_code: str) -> Dict[str, float]:
        import torch

        with torch.no_grad():
            emb = self._embed(cropped_code).unsqueeze(0)  # (1, H)
            logits = self.head(emb)                        # (1, 4)
            probs = torch.sigmoid(logits).squeeze(0)        # (4,)

        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        # Clip before scaling so max_boost is a hard cap.
        probs_np = np.clip(probs_np, 0.0, 1.0) * self.max_boost
        return {cat: float(probs_np[i]) for i, cat in enumerate(self._categories)}

    def error_boost(self, error_type: str) -> Dict[str, float]:
        return error_boost(error_type)

    # ---- persistence -------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the head state dict to disk (encoder stays frozen and shared)."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state_dict": self.head.state_dict(),
                "hidden_size": self.head.hidden_size,
                "num_categories": self.head.num_categories,
                "max_boost": self.max_boost,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        encoder: Any = None,
        tokenizer: Any = None,
        device: Optional[str] = None,
    ) -> "LearnedTypeRouter":
        """Load a :class:`LearnedTypeRouter` from a saved checkpoint."""
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        payload = torch.load(Path(path), map_location=device)
        head = LearnedRouterHead(
            hidden_size=int(payload.get("hidden_size", 768)),
            num_categories=int(payload.get("num_categories", 4)),
        )
        head.load_state_dict(payload["head_state_dict"])
        return cls(
            encoder=encoder,
            tokenizer=tokenizer,
            head=head,
            max_boost=float(payload.get("max_boost", 0.2)),
            device=device,
        )


class LearnedRouterHead(nn.Module):
    """Classifier head used by :class:`LearnedTypeRouter`.

    Architecture:
        ``Linear(hidden_size, num_categories)`` followed by scaling by a
        learned scalar temperature (``log_temperature.exp()``). Returned
        logits are fed to ``torch.sigmoid`` by :class:`LearnedTypeRouter` to
        produce per-category probabilities.

    The temperature lets the model sharpen or flatten the boost distribution
    during training; it is initialised at 1.0 (``log_temperature = 0``).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_categories = int(num_categories)
        self.linear = nn.Linear(self.hidden_size, self.num_categories)
        self.log_temperature = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        logits = self.linear(emb)
        return logits / self.log_temperature.exp()
