"""
data_pipeline.py — Contrastive triplet generation for HCCS training.

For each RepoBench v1.1 example, the ``gold_snippet_index`` identifies which
context chunk is needed to correctly predict ``next_line``.  Triplets are
formed deterministically:

    positive = context[gold_snippet_index]
    negatives = all other context chunks

No code execution is required — labels come directly from the dataset.

Target: ~70K–80K triplets from 8,033 examples.

The full data pipeline runs in notebooks/01_data_pipeline.ipynb.
This module provides the data structures and serialisation helpers.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveTriplet:
    """A training example for the HCCS scorer.

    Attributes:
        query:              The code written so far (``cropped_code`` from RepoBench).
        positive_context:   Snippet from the gold context chunk.
        negative_context:   Snippet from a non-gold context chunk.
        positive_path:      File path of the gold context chunk.
        negative_path:      File path of the negative context chunk.
        task_id:            Identifier: ``"{repo_name}::{file_path}::{idx}"``.
        gold_snippet_index: Index into the example's ``context[]`` list.
    """

    query: str
    positive_context: str
    negative_context: str
    positive_path: str
    negative_path: str
    task_id: str
    gold_snippet_index: int


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _triplet_to_dict(triplet: ContrastiveTriplet) -> Dict[str, Any]:
    """Convert a triplet to a JSON-serialisable dict."""
    return asdict(triplet)


def _dict_to_triplet(d: Dict[str, Any]) -> ContrastiveTriplet:
    """Reconstruct a ContrastiveTriplet from a dict (e.g. parsed from JSONL)."""
    return ContrastiveTriplet(
        query=d["query"],
        positive_context=d["positive_context"],
        negative_context=d["negative_context"],
        positive_path=d["positive_path"],
        negative_path=d["negative_path"],
        task_id=d["task_id"],
        gold_snippet_index=d["gold_snippet_index"],
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
# Triplet generation from RepoBench examples
# ---------------------------------------------------------------------------

def create_triplets_from_example(
    example: Dict[str, Any],
    idx: int,
    max_negatives: Optional[int] = None,
    seed: int = 42,
) -> List[ContrastiveTriplet]:
    """Generate contrastive triplets from a single RepoBench example.

    The gold chunk (``context[gold_snippet_index]``) is paired with every
    other chunk to form (positive, negative) triplets.  If ``max_negatives``
    is set, a random subset of negatives is sampled instead.

    Args:
        example:       A single RepoBench example dict with keys:
                       ``repo_name``, ``file_path``, ``cropped_code``,
                       ``context`` (list of dicts with ``snippet``, ``path``,
                       ``identifier``), ``gold_snippet_index``.
        idx:           Dataset index (used to build ``task_id``).
        max_negatives: Cap on negative samples per example.  If None, use all.
        seed:          Random seed for negative sampling.

    Returns:
        List of ``ContrastiveTriplet`` instances.  Empty if
        ``gold_snippet_index`` is out of range or context has < 2 chunks.
    """
    context = example["context"]
    gold_idx = example["gold_snippet_index"]

    # Validate gold index
    if gold_idx < 0 or gold_idx >= len(context):
        return []
    if len(context) < 2:
        return []

    query = example["cropped_code"]
    task_id = f"{example['repo_name']}::{example['file_path']}::{idx}"
    gold_chunk = context[gold_idx]
    positive_snippet = gold_chunk["snippet"]
    positive_path = gold_chunk["path"]

    # Collect negative indices
    neg_indices = [i for i in range(len(context)) if i != gold_idx]

    if max_negatives is not None and len(neg_indices) > max_negatives:
        rng = random.Random(seed + idx)
        neg_indices = rng.sample(neg_indices, max_negatives)

    triplets: List[ContrastiveTriplet] = []
    for neg_idx in neg_indices:
        neg_chunk = context[neg_idx]
        triplets.append(
            ContrastiveTriplet(
                query=query,
                positive_context=positive_snippet,
                negative_context=neg_chunk["snippet"],
                positive_path=positive_path,
                negative_path=neg_chunk["path"],
                task_id=task_id,
                gold_snippet_index=gold_idx,
            )
        )

    return triplets


def split_dataset_by_repo(
    dataset: List[Dict[str, Any]], 
    train_ratio: float = 0.8, 
    seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits the dataset by repository name to prevent data leakage.
    All triplets derived from the same repo will stay in the same split.
    """
    import random
    
    # Identify unique repositories
    repos = sorted(list(set(ex['repo_name'] for ex in dataset)))
    random.Random(seed).shuffle(repos)
    
    split_idx = int(len(repos) * train_ratio)
    train_repos = set(repos[:split_idx])
    
    train_examples = [ex for ex in dataset if ex['repo_name'] in train_repos]
    val_examples = [ex for ex in dataset if ex['repo_name'] not in train_repos]
    
    return train_examples, val_examples


def create_all_triplets(
    dataset: Any,
    max_negatives: Optional[int] = None,
    seed: int = 42,
) -> List[ContrastiveTriplet]:
    """Generate contrastive triplets from every example in a RepoBench dataset.

    Args:
        dataset:       A HuggingFace ``Dataset`` (or list of dicts) from
                       ``load_dataset("tianyang/repobench_python_v1.1",
                       split="cross_file_first")``.
        max_negatives: Cap on negative samples per example.  If None, use all.
        seed:          Random seed for negative sampling.

    Returns:
        List of all ``ContrastiveTriplet`` instances across the dataset.
    """
    all_triplets: List[ContrastiveTriplet] = []

    for idx, example in enumerate(dataset):
        triplets = create_triplets_from_example(
            example, idx, max_negatives=max_negatives, seed=seed
        )
        all_triplets.extend(triplets)

    return all_triplets


# ---------------------------------------------------------------------------
# Dataset statistics helper
# ---------------------------------------------------------------------------

def summarise_triplets(triplets: List[ContrastiveTriplet]) -> Dict[str, Any]:
    """Compute summary statistics over a list of triplets.

    Args:
        triplets: List of ``ContrastiveTriplet`` instances.

    Returns:
        Dict with keys: ``total``, ``unique_tasks`` (number of distinct tasks),
        ``avg_negatives_per_task`` (average triplets per unique task).
    """
    task_counts: Dict[str, int] = {}
    for t in triplets:
        task_counts[t.task_id] = task_counts.get(t.task_id, 0) + 1

    n_tasks = len(task_counts)
    avg_neg = sum(task_counts.values()) / n_tasks if n_tasks > 0 else 0.0

    return {
        "total": len(triplets),
        "unique_tasks": n_tasks,
        "avg_negatives_per_task": round(avg_neg, 1),
    }
