"""
data_pipeline.py — Contrastive triplet generation for HCCS training.

For each coding task in CodeHaluEval:
  1. Sample 6 different context subsets from the repo chunks.
  2. Generate code with each subset using the LLM.
  3. Execute each generated code against test cases.
  4. Pair a PASSING subset with a FAILING subset → contrastive triplet.

Target: ~3,000 usable triplets from ~500 tasks.

The full generation loop runs in notebooks/01_data_pipeline.ipynb.
This module provides the data structures and serialisation helpers.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from haluguard.chunker import chunk_repo
from haluguard.efl import ExecutionResult, execute_code
from haluguard.hccs import HallucinationType


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveTriplet:
    """A training example for the HCCS scorer.

    Attributes:
        query:            Natural-language coding task description.
        positive_context: Context string that led to PASSING code.
        negative_context: Context string that led to FAILING (hallucinatory) code.
        hallucination_type: Category of the failure in the negative example.
        task_id:          Identifier from the source dataset (e.g. CodeHaluEval).
    """

    query: str
    positive_context: str
    negative_context: str
    hallucination_type: Optional[HallucinationType]
    task_id: str


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _triplet_to_dict(triplet: ContrastiveTriplet) -> Dict[str, Any]:
    """Convert a triplet to a JSON-serialisable dict."""
    d = asdict(triplet)
    if triplet.hallucination_type is not None:
        d["hallucination_type"] = triplet.hallucination_type.value
    return d


def _dict_to_triplet(d: Dict[str, Any]) -> ContrastiveTriplet:
    """Reconstruct a ContrastiveTriplet from a dict (e.g. parsed from JSONL)."""
    hall_type: Optional[HallucinationType] = None
    raw = d.get("hallucination_type")
    if raw is not None:
        try:
            hall_type = HallucinationType(raw)
        except ValueError:
            pass

    return ContrastiveTriplet(
        query=d["query"],
        positive_context=d["positive_context"],
        negative_context=d["negative_context"],
        hallucination_type=hall_type,
        task_id=d["task_id"],
    )


def save_triplets(triplets: List[ContrastiveTriplet], path: Path) -> None:
    """Save a list of triplets to a JSONL file (one JSON object per line).

    Creates parent directories if they do not exist.

    Args:
        triplets: List of ``ContrastiveTriplet`` instances.
        path:     Destination ``.jsonl`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(_triplet_to_dict(triplet)) + "\n")


def load_triplets(path: Path) -> List[ContrastiveTriplet]:
    """Load triplets from a JSONL file written by ``save_triplets``.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of ``ContrastiveTriplet`` instances.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    triplets: List[ContrastiveTriplet] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(_dict_to_triplet(json.loads(line)))

    return triplets


# ---------------------------------------------------------------------------
# Context subset sampling
# ---------------------------------------------------------------------------

def generate_context_subsets(
    chunks: List[str],
    n_subsets: int = 6,
    min_chunks: int = 1,
    max_chunks: int = 5,
    seed: Optional[int] = None,
) -> List[List[str]]:
    """Sample random subsets of context chunks for contrastive data generation.

    Draws ``n_subsets`` subsets of varying sizes.  Each subset is a random
    sample of chunks from the full chunk list, so different subsets give the
    LLM different amounts and types of context.

    Args:
        chunks:     Full list of chunks from ``chunk_repo``.
        n_subsets:  Number of subsets to generate.  Default 6.
        min_chunks: Minimum chunks per subset.  Default 1.
        max_chunks: Maximum chunks per subset (capped by len(chunks)).
        seed:       Random seed for reproducibility.

    Returns:
        List of ``n_subsets`` chunk lists.  May contain duplicates if the
        chunk pool is small.
    """
    rng = random.Random(seed)
    max_chunks = min(max_chunks, len(chunks))
    subsets: List[List[str]] = []

    for _ in range(n_subsets):
        k = rng.randint(min_chunks, max_chunks)
        subset = rng.sample(chunks, k)
        subsets.append(subset)

    return subsets


# ---------------------------------------------------------------------------
# Triplet creation (stub — full logic in notebooks/01_data_pipeline.ipynb)
# ---------------------------------------------------------------------------

def create_triplets_from_task(
    task_id: str,
    query: str,
    repo_files: Dict[str, str],
    test_code: str,
    generate_fn: Callable[[str], str],
    n_subsets: int = 6,
    seed: Optional[int] = None,
) -> List[ContrastiveTriplet]:
    """Generate contrastive triplets for a single coding task.

    For each of ``n_subsets`` context subsets:
      1. Build a prompt from the subset + query.
      2. Call ``generate_fn`` to produce a code candidate.
      3. Execute the candidate against ``test_code``.

    Then pair each PASSING subset with a FAILING subset to form triplets.

    Args:
        task_id:     Identifier for this task (from the source dataset).
        query:       Natural-language task description.
        repo_files:  Mapping of filepath → source for the target repo.
        test_code:   Test assertions used to judge generated code.
        generate_fn: Callable that takes a prompt and returns generated code.
        n_subsets:   Number of context subsets to try.  Default 6.
        seed:        Random seed for subset sampling.

    Returns:
        List of ``ContrastiveTriplet`` instances.  May be empty if all subsets
        produce the same outcome (all pass or all fail).
    """
    chunks = chunk_repo(repo_files)
    if not chunks:
        return []

    subsets = generate_context_subsets(
        chunks, n_subsets=n_subsets, seed=seed
    )

    # Collect (context_str, execution_result) for each subset
    outcomes: List[Tuple[str, Any]] = []
    for subset in subsets:
        context_str = "\n\n".join(subset)
        prompt = (
            "# Repository context:\n\n"
            + context_str
            + "\n\n# Task:\n# "
            + query
            + "\n\n# Write the implementation below:\n"
        )
        code = generate_fn(prompt)
        result = execute_code(code, test_code)
        outcomes.append((context_str, result))

    # Cross-pair each PASS with each FAIL → contrastive triplets
    passes = [
        (ctx, r) for ctx, r in outcomes if r.passed
    ]
    fails = [
        (ctx, r) for ctx, r in outcomes if not r.passed
    ]

    triplets: List[ContrastiveTriplet] = []
    for pos_ctx, _ in passes:
        for neg_ctx, neg_result in fails:
            triplets.append(
                ContrastiveTriplet(
                    query=query,
                    positive_context=pos_ctx,
                    negative_context=neg_ctx,
                    hallucination_type=neg_result.hallucination_type,
                    task_id=task_id,
                )
            )

    return triplets


# ---------------------------------------------------------------------------
# Dataset statistics helper
# ---------------------------------------------------------------------------

def summarise_triplets(triplets: List[ContrastiveTriplet]) -> Dict[str, Any]:
    """Compute summary statistics over a list of triplets.

    Args:
        triplets: List of ``ContrastiveTriplet`` instances.

    Returns:
        Dict with keys: ``total``, ``by_hallucination_type`` (counts per type),
        and ``unknown`` (triplets with no hallucination type label).
    """
    by_type: Dict[str, int] = {h.value: 0 for h in HallucinationType}
    unknown = 0

    for t in triplets:
        if t.hallucination_type is None:
            unknown += 1
        else:
            by_type[t.hallucination_type.value] += 1

    return {
        "total": len(triplets),
        "by_hallucination_type": by_type,
        "unknown": unknown,
    }
