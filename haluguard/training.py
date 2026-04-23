"""
training.py — Model-agnostic training utilities for HCCS scorers.

These helpers keep training logic out of notebooks so the ranking objective,
negative sampling, and retrieval metrics are testable and reusable across all
model architectures in haluguard/models.py.

Key additions over the original version:
- ``score_with_model``   : dispatches to any scorer's forward() signature
- ``curriculum_train_epoch``: easy → hard negative progression
- ``in_batch_negative_loss``: uses other examples in the batch as extra negatives
- ``focal_loss``         : down-weights easy, well-ranked examples
- ``swa_update``         : wrapper for stochastic weight averaging
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from haluguard.hccs import HallucinationType, HCCSScorer, build_pair_features, embed_code
from haluguard.type_router import (
    ERROR_TO_CATEGORY,
    LearnedRouterHead,
    LearnedTypeRouter,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RankingExample:
    """One repository-level ranking instance used for HCCS training.

    Attributes:
        example_id: Stable index into the RepoBench split.
        query_emb: Query embedding for ``cropped_code``. Shape ``(emb_dim,)``.
        chunk_embs: All candidate chunk embeddings. Shape ``(n_chunks, emb_dim)``.
        gold_index: Index of the gold chunk inside ``chunk_embs``.
        hard_negative_indices: Non-gold chunk indices sorted by descending
            cosine similarity to the query embedding (hardest first).
    """

    example_id: int
    query_emb: Tensor
    chunk_embs: Tensor
    gold_index: int
    hard_negative_indices: List[int]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _normalise_embeddings(embeddings: Tensor, dim: int) -> Tensor:
    """L2-normalise embeddings with a numerical stability epsilon."""
    denom = embeddings.norm(dim=dim, keepdim=True).clamp(min=1e-8)
    return embeddings / denom


# ---------------------------------------------------------------------------
# Building RankingExamples from pre-computed embeddings
# ---------------------------------------------------------------------------

def build_ranking_examples(
    query_embs: Tensor,
    chunk_embs: Sequence[Tensor],
    gold_indices: Sequence[int],
) -> List[RankingExample]:
    """Build example-level ranking data from pre-computed RepoBench embeddings.

    Invalid examples (gold index out of range or fewer than 2 chunks) are
    silently skipped.

    Args:
        query_embs:  Shape ``(n_examples, emb_dim)``.
        chunk_embs:  Per-example list of tensors, each ``(n_chunks_i, emb_dim)``.
        gold_indices: Gold chunk index for each example.

    Returns:
        List of :class:`RankingExample` instances (may be shorter than input).
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


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

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
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Create dataset-level 70/15/15 train/val/test index splits.

    Args:
        n_examples:  Total number of examples in the dataset.
        train_ratio: Fraction allocated to training.  Default 0.7.
        val_ratio:   Fraction allocated to validation.  Default 0.15.
        seed:        Random seed for reproducibility.  Default 42.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``, each a sorted list
        of integer indices.
    """
    indices = list(range(n_examples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(n_examples * train_ratio)
    val_end = train_end + int(n_examples * val_ratio)

    return {
        "train": sorted(indices[:train_end]),
        "val":   sorted(indices[train_end:val_end]),
        "test":  sorted(indices[val_end:]),
    }


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def select_training_chunk_indices(
    example: RankingExample,
    num_hard_negatives: int = 4,
    num_random_negatives: int = 3,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Select gold + hard negatives + random negatives for listwise training.

    Returns a list starting with the gold index (always at position 0),
    followed by hard negatives and random negatives.
    """
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


# ---------------------------------------------------------------------------
# Datasets & Dataloaders
# ---------------------------------------------------------------------------

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
    """Create training and validation dataloaders for listwise ranking."""
    if use_all_chunks_for_training:
        train_dataset: Dataset = ListwiseEvalDataset(train_examples)
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


# ---------------------------------------------------------------------------
# Model-agnostic forward dispatch
# ---------------------------------------------------------------------------

def score_with_model(
    model: nn.Module,
    query_embs: Tensor,
    chunk_embs: Tensor,
) -> Tensor:
    """Compute per-chunk logits for any scorer architecture.

    Dispatches based on the model type:
    - Models with a ``forward(query_embs, chunk_embs)`` signature (all models
      in haluguard.models) are called directly.
    - The legacy ``HCCSScorer`` uses ``build_pair_features`` + ``forward_logits``.

    Args:
        model:      Any scorer from haluguard.models or haluguard.hccs.
        query_embs: ``(B, emb_dim)``
        chunk_embs: ``(B, max_chunks, emb_dim)``

    Returns:
        Logits of shape ``(B, max_chunks)``.
    """
    if isinstance(model, HCCSScorer):
        # Legacy path: HCCSScorer expects 4× interaction features
        batch_size, max_chunks, _ = chunk_embs.shape
        repeated = query_embs.unsqueeze(1).expand(-1, max_chunks, -1)
        pair_features = build_pair_features(repeated, chunk_embs)
        flat_logits = model.forward_logits(
            pair_features.reshape(batch_size * max_chunks, -1)
        )
        return flat_logits.view(batch_size, max_chunks)

    # All models in haluguard.models expose forward(query_embs, chunk_embs)
    return model(query_embs, chunk_embs)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def masked_listwise_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy over valid chunk positions only.

    Args:
        logits:           ``(B, max_chunks)`` raw scores.
        targets:          ``(B,)`` gold chunk indices.
        mask:             ``(B, max_chunks)`` boolean mask (True = valid chunk).
        label_smoothing:  Smoothing factor for cross-entropy.  Default 0.0.

    Returns:
        Scalar loss.
    """
    masked_logits = logits.masked_fill(~mask, -1e9)
    return F.cross_entropy(masked_logits, targets, label_smoothing=label_smoothing)


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Focal loss: down-weights well-ranked (easy) examples.

    Computes standard cross-entropy then multiplies by ``(1 - p_t)^gamma``
    where ``p_t`` is the model's probability assigned to the correct class.
    High gamma (e.g. 2.0) focuses training on hard, misclassified examples.

    Args:
        logits:          ``(B, max_chunks)`` raw scores.
        targets:         ``(B,)`` gold chunk indices.
        mask:            ``(B, max_chunks)`` boolean mask.
        gamma:           Focusing parameter.  Default 2.0.
        label_smoothing: Smoothing factor.  Default 0.0.

    Returns:
        Scalar focal loss.
    """
    masked_logits = logits.masked_fill(~mask, -1e9)
    ce = F.cross_entropy(masked_logits, targets, reduction="none",
                         label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def in_batch_negative_loss(
    query_embs: Tensor,
    gold_chunk_embs: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """InfoNCE loss using other examples' gold chunks as in-batch negatives.

    For a batch of B examples, each query is contrasted against all B gold
    chunks (1 positive + B-1 in-batch negatives per query).  Requires that
    the model architecture uses L2-normalised representations — only suitable
    for DualEncoder / DualEncoderDeep.

    Args:
        query_embs:      L2-normalised query projections, ``(B, proj_dim)``.
        gold_chunk_embs: L2-normalised gold chunk projections, ``(B, proj_dim)``.
        temperature:     InfoNCE temperature.  Default 0.07.

    Returns:
        Scalar loss (mean over batch).
    """
    logits = torch.mm(query_embs, gold_chunk_embs.T) / temperature  # (B, B)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Training epochs
# ---------------------------------------------------------------------------

def train_listwise_epoch(
    scorer: nn.Module,
    dataloader: DataLoader,
    optimizer: Any,
    device: str,
    max_grad_norm: Optional[float] = 1.0,
    label_smoothing: float = 0.0,
    emb_dropout: float = 0.0,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    verbose: bool = False,
    desc: str = "train",
) -> float:
    """Train any scorer for one epoch with masked listwise ranking loss.

    Args:
        scorer:          Any model from haluguard.models or haluguard.hccs.
        dataloader:      Yields batches from :class:`ListwiseTrainDataset`.
        optimizer:       PyTorch optimiser.
        device:          Torch device string.
        max_grad_norm:   Gradient clipping norm.  None disables clipping.
        label_smoothing: Passed to cross-entropy.  Default 0.0.
        emb_dropout:     Randomly zeros embedding dimensions as augmentation.
        use_focal:       If True, use focal loss instead of cross-entropy.
        focal_gamma:     Focusing parameter for focal loss.  Default 2.0.

    Returns:
        Mean per-example training loss for this epoch.
    """
    scorer.train()
    total_loss = 0.0
    total_examples = 0

    iterator = dataloader
    if verbose:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(dataloader, desc=desc, leave=False)
        except ImportError:
            iterator = dataloader

    for batch in iterator:
        query_embs = batch["query_embs"].to(device)
        chunk_embs = batch["chunk_embs"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        # Optional embedding dropout (augmentation on pre-computed embeddings)
        if emb_dropout > 0.0 and scorer.training:
            query_embs = F.dropout(query_embs, p=emb_dropout)
            chunk_embs = F.dropout(chunk_embs, p=emb_dropout)

        logits = score_with_model(scorer, query_embs, chunk_embs)

        if use_focal:
            loss = focal_loss(logits, targets, mask, gamma=focal_gamma,
                              label_smoothing=label_smoothing)
        else:
            loss = masked_listwise_loss(logits, targets, mask,
                                        label_smoothing=label_smoothing)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(scorer.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = int(query_embs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def curriculum_train_epoch(
    scorer: nn.Module,
    train_examples: List[RankingExample],
    optimizer: Any,
    device: str,
    epoch: int,
    warmup_epochs: int = 5,
    batch_size: int = 256,
    max_hard_negatives: int = 4,
    max_grad_norm: Optional[float] = 1.0,
    label_smoothing: float = 0.0,
    emb_dropout: float = 0.0,
    seed: int = 42,
    verbose: bool = False,
) -> float:
    """Curriculum training: easy negatives first, gradually harder.

    During the first ``warmup_epochs`` epochs only random negatives are used.
    Afterwards, hard negatives are progressively introduced, reaching
    ``max_hard_negatives`` at ``2 * warmup_epochs``.

    Args:
        scorer:            Any scorer from haluguard.models.
        train_examples:    List of :class:`RankingExample`.
        optimizer:         PyTorch optimiser.
        device:            Torch device string.
        epoch:             Current epoch index (0-based).
        warmup_epochs:     Epochs to use only easy (random) negatives.
        batch_size:        Number of examples per gradient step.
        max_hard_negatives: Maximum hard negatives after warmup.
        max_grad_norm:     Gradient clipping norm.
        label_smoothing:   CE label smoothing.
        emb_dropout:       Embedding dropout probability.
        seed:              RNG seed (offset by epoch for shuffling).

    Returns:
        Mean per-example training loss for this epoch.
    """
    # Determine how many hard negatives to use this epoch
    if epoch < warmup_epochs:
        n_hard = 0
    else:
        progress = min((epoch - warmup_epochs) / max(warmup_epochs, 1), 1.0)
        n_hard = int(progress * max_hard_negatives)

    n_random = max(1, 7 - n_hard)   # total negatives ≈ 7 throughout

    dataset = ListwiseTrainDataset(
        train_examples,
        num_hard_negatives=n_hard,
        num_random_negatives=n_random,
        seed=seed + epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ranking_batch,
    )
    return train_listwise_epoch(
        scorer, loader, optimizer, device,
        max_grad_norm=max_grad_norm,
        label_smoothing=label_smoothing,
        emb_dropout=emb_dropout,
        verbose=verbose,
        desc=f"epoch {epoch} (hard={n_hard}, rand={n_random})",
    )


# ---------------------------------------------------------------------------
# Legacy helper (keeps backwards compatibility with score_listwise_logits calls)
# ---------------------------------------------------------------------------

def score_listwise_logits(
    scorer: nn.Module,
    query_embs: Tensor,
    chunk_embs: Tensor,
) -> Tensor:
    """Score a batch of (query, many chunks) examples as raw logits.

    Alias for ``score_with_model`` retained for backwards compatibility.
    """
    return score_with_model(scorer, query_embs, chunk_embs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    top_ks: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Compute retrieval metrics for a validation batch.

    Args:
        logits:  ``(B, max_chunks)`` raw scores.
        targets: ``(B,)`` gold chunk indices.
        mask:    ``(B, max_chunks)`` boolean validity mask.
        top_ks:  List of K values.  Default ``[1, 3, 5]``.

    Returns:
        Dict with keys ``"mrr"``, ``"acc@k"`` / ``"recall@k"`` for each K.
    """
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


def evaluate_listwise_model(
    scorer: nn.Module,
    dataloader: DataLoader,
    device: str,
    verbose: bool = False,
) -> Dict[str, float]:
    """Evaluate any scorer with listwise loss + retrieval metrics.

    Args:
        scorer:     Any model from haluguard.models or haluguard.hccs.
        dataloader: Yields batches from :class:`ListwiseEvalDataset`.
        device:     Torch device string.

    Returns:
        Dict with keys ``"loss"``, ``"mrr"``, ``"recall@1"``, ``"recall@3"``,
        ``"recall@5"``, ``"acc@1"``, ``"acc@3"``, ``"acc@5"``.
    """
    scorer.eval()
    total_loss = 0.0
    total_examples = 0
    metric_names = ["acc@1", "acc@3", "acc@5", "mrr"]
    metric_sums = {name: 0.0 for name in metric_names}

    iterator = dataloader
    if verbose:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(dataloader, desc="eval", leave=False)
        except ImportError:
            iterator = dataloader

    with torch.no_grad():
        for batch in iterator:
            query_embs = batch["query_embs"].to(device)
            chunk_embs = batch["chunk_embs"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            logits = score_with_model(scorer, query_embs, chunk_embs)
            loss = masked_listwise_loss(logits, targets, mask)
            metrics = compute_retrieval_metrics(logits, targets, mask)

            batch_size = int(query_embs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            for name in metric_names:
                metric_sums[name] += metrics[name] * batch_size

    result: Dict[str, float] = {
        "loss": total_loss / max(total_examples, 1),
    }
    for name in metric_names:
        result[name] = metric_sums[name] / max(total_examples, 1)

    # Backwards-compatible aliases
    result["recall@1"] = result["acc@1"]
    result["recall@3"] = result["acc@3"]
    result["recall@5"] = result["acc@5"]
    return result


# ---------------------------------------------------------------------------
# Learned type router (multi-label classifier over HallucinationType)
# ---------------------------------------------------------------------------

@dataclass
class RouterTrace:
    """A single example used to train :class:`LearnedTypeRouter`.

    Attributes:
        cropped_code: Query code (what the EFL saw on iteration 1).
        error_type:   First-iteration Python exception class name, or ``None``
                      if the example passed (→ all-zero target label).
    """

    cropped_code: str
    error_type: Optional[str]


def _category_index() -> Dict[str, int]:
    return {h.value: i for i, h in enumerate(HallucinationType)}


def _trace_to_label(
    trace: RouterTrace,
    cat_index: Dict[str, int],
    num_categories: int,
) -> torch.Tensor:
    """Multi-label one-hot from a router trace (zero vector if passed)."""
    y = torch.zeros(num_categories, dtype=torch.float32)
    if trace.error_type is None:
        return y
    category = ERROR_TO_CATEGORY.get(trace.error_type)
    if category is None:
        return y
    idx = cat_index.get(category)
    if idx is not None:
        y[idx] = 1.0
    return y


def _embed_traces(
    traces: Sequence[RouterTrace],
    tokenizer: Any,
    encoder: Any,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Embed every trace's ``cropped_code`` with a frozen encoder."""
    vecs: List[torch.Tensor] = []
    for trace in traces:
        emb = embed_code(
            trace.cropped_code,
            tokenizer=tokenizer,
            model=encoder,
            device=device,
            truncation_side="left",
        )
        vecs.append(torch.as_tensor(emb, dtype=torch.float32))
    return torch.stack(vecs, dim=0) if vecs else torch.empty(0, 768)


def train_learned_router(
    traces: Sequence[RouterTrace],
    encoder: Any,
    tokenizer: Any,
    hidden_size: int = 768,
    max_boost: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    device: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
) -> LearnedTypeRouter:
    """Train a :class:`LearnedTypeRouter` on a list of EFL traces.

    Architecture: frozen ``encoder`` → ``LearnedRouterHead`` (linear +
    log-temperature). Loss: binary cross-entropy with logits (multi-label).

    Args:
        traces:         EFL traces from
                        ``haluguard.run_eval`` / notebook 08 (cell 4).
        encoder:        Frozen CodeBERT-style encoder (kept in eval mode).
        tokenizer:      Matching tokenizer.
        hidden_size:    Encoder hidden dimension.
        max_boost:      Upper cap propagated to the returned router.
        lr:             Optimiser learning rate.
        weight_decay:   AdamW weight decay.
        epochs:         Maximum epochs; stops early on val-loss plateau.
        batch_size:     Training batch size.
        val_fraction:   Fraction of traces held out for validation.
        device:         Torch device string; inferred if omitted.
        seed:           RNG seed for train/val shuffle.
        verbose:        Print per-epoch loss/val-loss.

    Returns:
        A ready-to-use :class:`LearnedTypeRouter` wrapping the trained head
        (and the same frozen encoder).
    """
    if not traces:
        raise ValueError("train_learned_router: at least one trace is required")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    num_categories = len(HallucinationType)
    cat_index = _category_index()

    # Shuffle deterministically; split into train/val.
    rng = random.Random(seed)
    idxs = list(range(len(traces)))
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(idxs) * val_fraction))) if len(idxs) > 1 else 0
    val_idxs = set(idxs[:n_val])
    train_traces = [traces[i] for i in idxs if i not in val_idxs]
    val_traces = [traces[i] for i in idxs if i in val_idxs]

    # One-off embedding pass (frozen encoder → no need to re-tokenise each epoch).
    X_train = _embed_traces(train_traces, tokenizer, encoder, device).to(device)
    y_train = torch.stack(
        [_trace_to_label(t, cat_index, num_categories) for t in train_traces]
    ).to(device) if train_traces else torch.empty(0, num_categories, device=device)

    if val_traces:
        X_val = _embed_traces(val_traces, tokenizer, encoder, device).to(device)
        y_val = torch.stack(
            [_trace_to_label(t, cat_index, num_categories) for t in val_traces]
        ).to(device)
    else:
        X_val = None
        y_val = None

    head = LearnedRouterHead(hidden_size=hidden_size, num_categories=num_categories).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = 2

    for epoch in range(1, epochs + 1):
        head.train()
        # Shuffle train indices each epoch.
        perm = torch.randperm(X_train.shape[0], device=device)
        total_loss = 0.0
        n_seen = 0
        for start in range(0, X_train.shape[0], batch_size):
            batch_idx = perm[start : start + batch_size]
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            logits = head(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().item()) * xb.shape[0]
            n_seen += xb.shape[0]
        train_loss = total_loss / max(n_seen, 1)

        if X_val is not None and y_val is not None:
            head.eval()
            with torch.no_grad():
                val_logits = head(X_val)
                val_loss = float(loss_fn(val_logits, y_val).item())
            if verbose:
                print(
                    f"[learned_router] epoch {epoch}/{epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )
            if val_loss + 1e-5 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                patience_left = 2
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if verbose:
                        print("[learned_router] early-stopping on val plateau")
                    break
        else:
            if verbose:
                print(f"[learned_router] epoch {epoch}/{epochs} train_loss={train_loss:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    return LearnedTypeRouter(
        encoder=encoder,
        tokenizer=tokenizer,
        head=head,
        max_boost=max_boost,
        device=device,
    )
