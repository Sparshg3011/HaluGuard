"""
training.py - Listwise training utilities for the HCCS scorer.

These helpers keep the training logic out of the Colab notebook so the
selection objective, hard-negative sampling, and retrieval metrics are easier
to test and reuse.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import numpy as np

from haluguard.baselines import blend_scores, cosine_scores
from haluguard.hccs import HCCSScorer, build_pair_features
from haluguard.retrieval_benchmark import compute_accuracy_metrics


@dataclass
class RankingExample:
    """One repository-level ranking instance used for HCCS training.

    Attributes:
        example_id: Stable index into the RepoBench split.
        query_emb: Query embedding for ``cropped_code``.
        chunk_embs: All candidate chunk embeddings for this query.
        gold_index: Index of the gold chunk inside ``chunk_embs``.
        hard_negative_indices: Non-gold chunk indices sorted by descending
            cosine similarity to the query embedding.
    """

    example_id: int
    query_emb: Tensor
    chunk_embs: Tensor
    gold_index: int
    hard_negative_indices: List[int]


def _normalise_embeddings(embeddings: Tensor, dim: int) -> Tensor:
    """L2-normalise embeddings with an epsilon for numerical stability."""
    denom = embeddings.norm(dim=dim, keepdim=True).clamp(min=1e-8)
    return embeddings / denom


def build_ranking_examples(
    query_embs: Tensor,
    chunk_embs: Sequence[Tensor],
    gold_indices: Sequence[int],
) -> List[RankingExample]:
    """Build example-level ranking data from pre-computed RepoBench embeddings.

    Invalid examples are skipped when the gold index is out of range or there
    are fewer than two chunks.
    """
    examples: List[RankingExample] = []

    for example_id, gold_index in enumerate(gold_indices):
        query_emb = query_embs[example_id].detach().cpu().float()
        example_chunk_embs = chunk_embs[example_id].detach().cpu().float()
        n_chunks = int(example_chunk_embs.shape[0])

        if gold_index < 0 or gold_index >= n_chunks or n_chunks < 2:
            continue

        query_norm = _normalise_embeddings(query_emb.unsqueeze(0), dim=1).squeeze(0)
        chunk_norms = _normalise_embeddings(example_chunk_embs, dim=1)
        similarities = torch.mv(chunk_norms, query_norm)

        negative_indices = [i for i in range(n_chunks) if i != int(gold_index)]
        hard_negative_indices = sorted(
            negative_indices,
            key=lambda idx: float(similarities[idx].item()),
            reverse=True,
        )

        examples.append(
            RankingExample(
                example_id=example_id,
                query_emb=query_emb,
                chunk_embs=example_chunk_embs,
                gold_index=int(gold_index),
                hard_negative_indices=hard_negative_indices,
            )
        )

    return examples


def split_ranking_examples(
    examples: Sequence[RankingExample],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[RankingExample], List[RankingExample]]:
    """Split ranking examples at the example level to avoid triplet leakage."""
    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split_index = int(len(indices) * train_ratio)
    train_examples = [examples[i] for i in indices[:split_index]]
    val_examples = [examples[i] for i in indices[split_index:]]
    return train_examples, val_examples


def create_data_splits(
    n_examples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Create dataset-level index splits: train / val / test.

    Shuffles all indices with a fixed seed, then slices into three contiguous
    groups.  The test set is everything not in train or val (roughly 10%).
    Returns **sorted** index lists so downstream code can use them in order.

    Args:
        n_examples:  Total number of examples in the dataset.
        train_ratio: Fraction allocated to training.  Default 0.8.
        val_ratio:   Fraction allocated to validation.  Default 0.1.
        seed:        Random seed for reproducibility.  Default 42.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to a
        sorted list of integer indices.
    """
    indices = list(range(n_examples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(n_examples * train_ratio)
    val_end = train_end + int(n_examples * val_ratio)

    return {
        "train": sorted(indices[:train_end]),
        "val": sorted(indices[train_end:val_end]),
        "test": sorted(indices[val_end:]),
    }


def select_training_chunk_indices(
    example: RankingExample,
    num_hard_negatives: int = 4,
    num_random_negatives: int = 3,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Select gold + hard negatives + random negatives for listwise training."""
    if rng is None:
        rng = random

    hard_indices = example.hard_negative_indices[:num_hard_negatives]
    hard_index_set = set(hard_indices)
    remaining_indices = [
        idx for idx in example.hard_negative_indices if idx not in hard_index_set
    ]

    n_random = min(num_random_negatives, len(remaining_indices))
    random_indices = rng.sample(remaining_indices, n_random) if n_random > 0 else []

    return [example.gold_index] + hard_indices + random_indices


class ListwiseTrainDataset(Dataset):
    """Example-level dataset that samples hard negatives each time it is read."""

    def __init__(
        self,
        examples: Sequence[RankingExample],
        num_hard_negatives: int = 4,
        num_random_negatives: int = 3,
        seed: int = 42,
    ) -> None:
        self.examples = list(examples)
        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        example = self.examples[index]
        selected_indices = select_training_chunk_indices(
            example,
            num_hard_negatives=self.num_hard_negatives,
            num_random_negatives=self.num_random_negatives,
            rng=self.rng,
        )
        selected_chunk_embs = example.chunk_embs[selected_indices]

        return {
            "query_emb": example.query_emb,
            "chunk_embs": selected_chunk_embs,
            "gold_index": torch.tensor(0, dtype=torch.long),
        }


class ListwiseEvalDataset(Dataset):
    """Validation dataset that ranks against every chunk for each example."""

    def __init__(self, examples: Sequence[RankingExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        example = self.examples[index]
        return {
            "query_emb": example.query_emb,
            "chunk_embs": example.chunk_embs,
            "gold_index": torch.tensor(example.gold_index, dtype=torch.long),
        }


def collate_ranking_batch(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Pad variable-length chunk lists so they can be scored as a batch."""
    batch_size = len(batch)
    hidden_dim = int(batch[0]["query_emb"].shape[0])
    max_chunks = max(int(item["chunk_embs"].shape[0]) for item in batch)

    query_batch = torch.stack([item["query_emb"].float() for item in batch], dim=0)
    chunk_batch = torch.zeros(batch_size, max_chunks, hidden_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n_chunks = int(item["chunk_embs"].shape[0])
        chunk_batch[i, :n_chunks] = item["chunk_embs"].float()
        mask[i, :n_chunks] = True
        targets[i] = item["gold_index"]

    return {
        "query_embs": query_batch,
        "chunk_embs": chunk_batch,
        "mask": mask,
        "targets": targets,
    }


def build_ranking_dataloaders(
    train_examples: Sequence[RankingExample],
    val_examples: Sequence[RankingExample],
    train_batch_size: int = 256,
    eval_batch_size: int = 128,
    use_all_chunks_for_training: bool = False,
    num_hard_negatives: int = 4,
    num_random_negatives: int = 3,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for listwise ranking.

    When ``use_all_chunks_for_training`` is True, the train loader exposes the
    full candidate list for each example so the training objective exactly
    matches validation and downstream retrieval.
    """
    if use_all_chunks_for_training:
        train_dataset = ListwiseEvalDataset(train_examples)
    else:
        train_dataset = ListwiseTrainDataset(
            train_examples,
            num_hard_negatives=num_hard_negatives,
            num_random_negatives=num_random_negatives,
            seed=seed,
        )
    val_dataset = ListwiseEvalDataset(val_examples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_ranking_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
    )
    return train_loader, val_loader


def score_listwise_logits(
    scorer: HCCSScorer,
    query_embs: Tensor,
    chunk_embs: Tensor,
) -> Tensor:
    """Score a batch of ``(query, many chunks)`` examples as raw logits."""
    batch_size, max_chunks, _ = chunk_embs.shape
    repeated_queries = query_embs.unsqueeze(1).expand(-1, max_chunks, -1)
    pair_features = build_pair_features(repeated_queries, chunk_embs)
    flat_logits = scorer.forward_logits(pair_features.reshape(batch_size * max_chunks, -1))
    return flat_logits.view(batch_size, max_chunks)


def masked_listwise_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    """Cross-entropy over valid chunk positions only."""
    masked_logits = logits.masked_fill(~mask, -1e9)
    return F.cross_entropy(masked_logits, targets)


def sweep_cosine_hccs_blend_metrics(
    val_examples: Sequence[RankingExample],
    scorer: HCCSScorer,
    device: str,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate min-max blended scores ``w * HCCS + (1-w) * cosine`` on validation.

    For each weight ``w``, ``w`` applies to **HCCS** scores after per-example
    min-max normalization (see ``blend_scores``).  Use this to pick a blend
    weight that maximises ``acc@5`` or MRR without retraining.

    Args:
        val_examples: Ranking instances (same objects as listwise training).
        scorer:       Trained ``HCCSScorer`` in eval mode.
        device:       Torch device for HCCS forward passes.
        weights:      Blend weights on HCCS; default ``[0, 0.25, 0.5, 0.75, 1]``.

    Returns:
        One dict per weight with ``blend_weight_hccs``, ``n_examples``, and
        ``acc@1``, ``acc@3``, ``acc@5``, ``mrr``.
    """
    if weights is None:
        weights = [0.0, 0.25, 0.5, 0.75, 1.0]

    results: List[Dict[str, Any]] = []
    for w in weights:
        gold_ranks: List[int] = []
        for example in val_examples:
            query_np = example.query_emb.numpy()
            chunk_np = example.chunk_embs.numpy()
            if chunk_np.shape[0] == 0:
                continue
            cos = cosine_scores(query_np, chunk_np)
            hccs = scorer.score_chunks(query_np, chunk_np, device=device)
            blended = blend_scores(hccs, cos, float(w), normalize=True)
            order = np.argsort(blended)[::-1].tolist()
            gold_ranks.append(order.index(int(example.gold_index)) + 1)

        metrics = compute_accuracy_metrics(gold_ranks, top_ks=[1, 3, 5])
        results.append(
            {
                "blend_weight_hccs": float(w),
                "n_examples": len(gold_ranks),
                **metrics,
            }
        )

    return results


def compute_retrieval_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    top_ks: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Compute retrieval metrics for a validation batch."""
    if top_ks is None:
        top_ks = [1, 3, 5]

    masked_logits = logits.masked_fill(~mask, -1e9)
    ranked_indices = torch.argsort(masked_logits, dim=1, descending=True)

    hits = {int(k): 0 for k in top_ks}
    reciprocal_rank_sum = 0.0
    total_examples = int(targets.shape[0])

    for row_idx in range(total_examples):
        valid_count = int(mask[row_idx].sum().item())
        target_index = int(targets[row_idx].item())
        ranking = ranked_indices[row_idx, :valid_count]
        rank_position = int((ranking == target_index).nonzero(as_tuple=False)[0].item()) + 1

        reciprocal_rank_sum += 1.0 / rank_position
        for k in hits:
            if rank_position <= min(k, valid_count):
                hits[k] += 1

    metrics: Dict[str, float] = {
        "mrr": reciprocal_rank_sum / max(total_examples, 1),
    }
    for k in sorted(hits.keys()):
        value = hits[k] / max(total_examples, 1)
        metrics[f"acc@{k}"] = value
        metrics[f"recall@{k}"] = value
    return metrics


def train_listwise_epoch(
    scorer: HCCSScorer,
    dataloader: DataLoader,
    optimizer: Any,
    device: str,
    max_grad_norm: Optional[float] = None,
) -> float:
    """Train the HCCS scorer for one epoch with listwise ranking loss."""
    scorer.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        query_embs = batch["query_embs"].to(device)
        chunk_embs = batch["chunk_embs"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        logits = score_listwise_logits(scorer, query_embs, chunk_embs)
        loss = masked_listwise_loss(logits, targets, mask)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = int(query_embs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def evaluate_listwise_model(
    scorer: HCCSScorer,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate the HCCS scorer with loss + retrieval metrics."""
    scorer.eval()
    total_loss = 0.0
    total_examples = 0
    metric_names = ["acc@1", "acc@3", "acc@5", "mrr"]
    metric_sums = {name: 0.0 for name in metric_names}

    with torch.no_grad():
        for batch in dataloader:
            query_embs = batch["query_embs"].to(device)
            chunk_embs = batch["chunk_embs"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            logits = score_listwise_logits(scorer, query_embs, chunk_embs)
            loss = masked_listwise_loss(logits, targets, mask)
            metrics = compute_retrieval_metrics(logits, targets, mask)

            batch_size = int(query_embs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            for name in metric_names:
                metric_sums[name] += metrics[name] * batch_size

    result = {
        "loss": total_loss / max(total_examples, 1),
    }
    for name in metric_names:
        result[name] = metric_sums[name] / max(total_examples, 1)

    # Backwards-compatible aliases.
    result["recall@1"] = result["acc@1"]
    result["recall@3"] = result["acc@3"]
    result["recall@5"] = result["acc@5"]
    return result
