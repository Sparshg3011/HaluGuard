"""
hccs.py — Hallucination Context Contrast Scorer (HCCS).

Architecture:
    Frozen CodeBERT encoder (microsoft/codebert-base, 768-dim CLS token)
    + pair features [query, context, |query-context|, query*context]
    → Linear(3072, 512) → ReLU → Dropout(0.1) → Linear(512, 1) → Sigmoid

Training signal:
    InfoNCE contrastive loss over (query, positive_ctx, negative_ctx) triplets.
    The encoder is always frozen; only the MLP scorer weights are updated.

Key design note:
    HallucinationType is defined here (not in a separate types.py) to serve as
    the single source of truth imported by type_router, efl, and data_pipeline.
    This prevents circular imports.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared enum — imported by type_router, efl, data_pipeline, pipeline
# ---------------------------------------------------------------------------

class HallucinationType(enum.Enum):
    """Categories of code hallucination produced by LLMs.

    Each category corresponds to a family of Python exceptions and requires
    different remediation context (see type_router.py).
    """

    RESOURCE = "resource"  # ImportError, ModuleNotFoundError
    NAMING   = "naming"    # NameError, UnboundLocalError, AttributeError
    MAPPING  = "mapping"   # KeyError, IndexError, TypeError
    LOGIC    = "logic"     # ValueError, AssertionError, RuntimeError


# ---------------------------------------------------------------------------
# Embedding helper (frozen encoder)
# ---------------------------------------------------------------------------

def _tokenize_with_side(
    text_or_texts: Any,
    tokenizer: Any,
    max_length: int,
    truncation_side: Optional[str] = None,
) -> Dict[str, Tensor]:
    """Tokenize while temporarily overriding ``tokenizer.truncation_side``."""
    original_side = getattr(tokenizer, "truncation_side", None)
    if truncation_side is not None and original_side is not None:
        tokenizer.truncation_side = truncation_side

    try:
        return tokenizer(
            text_or_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
    finally:
        if truncation_side is not None and original_side is not None:
            tokenizer.truncation_side = original_side


def _normalise_embedding_array(embedding: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D or 2-D embedding array."""
    if embedding.ndim == 1:
        denom = max(float(np.linalg.norm(embedding)), 1e-8)
        return embedding / denom

    denom = np.linalg.norm(embedding, axis=1, keepdims=True)
    denom = np.clip(denom, a_min=1e-8, a_max=None)
    return embedding / denom


def _pool_last_hidden_state(
    last_hidden_state: Tensor,
    attention_mask: Optional[Tensor],
    pooling: str,
) -> Tensor:
    """Pool encoder outputs into a fixed-size embedding."""
    if pooling == "cls":
        return last_hidden_state[:, 0, :]

    if pooling == "mean":
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)
        weights = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp(min=1e-8)
        return summed / counts

    raise ValueError(f"Unsupported pooling mode: {pooling}")


def embed_code(
    text: str,
    tokenizer: Any,
    model: Any,
    device: Optional[str] = None,
    truncation_side: str = "left",
    max_length: int = 512,
    normalize: bool = False,
    pooling: str = "cls",
) -> np.ndarray:
    """Encode a code string to a fixed-size vector using the CLS token.

    Runs a frozen HuggingFace encoder (e.g. ``microsoft/codebert-base``) in
    no-grad mode.  The model must be in ``eval()`` mode and should NOT be
    updated by any optimizer.

    Args:
        text:      Source code or natural-language query string to embed.
        tokenizer: HuggingFace tokenizer matching the encoder.
        model:     Frozen HuggingFace encoder model in eval mode.
        device:    Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
                   Inferred from CUDA availability if None.
        truncation_side: Which side to truncate when ``text`` exceeds ``max_length``
                   tokens. Queries default to ``"left"`` so the latest code
                   near the prediction point is preserved.
        max_length: Maximum token length fed to the encoder.
        normalize:  When True, L2-normalise the returned embedding.
        pooling:    Pooling strategy for encoder outputs. ``"cls"`` uses the
                    first token embedding; ``"mean"`` uses attention-masked
                    mean pooling across tokens.

    Returns:
        1-D numpy array of shape ``(hidden_size,)``, e.g. ``(768,)`` for
        ``microsoft/codebert-base``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = _tokenize_with_side(
        text,
        tokenizer=tokenizer,
        max_length=max_length,
        truncation_side=truncation_side,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (batch=1, seq_len, hidden_size)
    # Index 0 → CLS token, the sentence-level representation
    pooled_emb = _pool_last_hidden_state(
        outputs.last_hidden_state,
        inputs.get("attention_mask"),
        pooling=pooling,
    )
    embedding = pooled_emb.squeeze(0).cpu().numpy()
    return _normalise_embedding_array(embedding) if normalize else embedding


def batch_embed(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    device: Optional[str] = None,
    batch_size: int = 32,
    truncation_side: str = "right",
    max_length: int = 512,
    normalize: bool = False,
    pooling: str = "cls",
) -> np.ndarray:
    """Embed a list of code strings in batches.

    More efficient than calling ``embed_code`` in a loop because it minimises
    the number of tokenizer and forward-pass calls.

    Args:
        texts:      List of source code / query strings.
        tokenizer:  HuggingFace tokenizer.
        model:      Frozen HuggingFace encoder in eval mode.
        device:     Torch device string.  Inferred if None.
        batch_size: Number of strings to process per forward pass.
        truncation_side: Which side to truncate when a text exceeds ``max_length``
                   tokens. Context chunks default to ``"right"`` so the start
                   of a definition is preserved.
        max_length: Maximum token length fed to the encoder.
        normalize:  When True, L2-normalise each returned embedding.
        pooling:    Pooling strategy for encoder outputs. ``"cls"`` uses the
                    first token embedding; ``"mean"`` uses attention-masked
                    mean pooling across tokens.

    Returns:
        2-D numpy array of shape ``(len(texts), hidden_size)``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_embs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _tokenize_with_side(
            batch,
            tokenizer=tokenizer,
            max_length=max_length,
            truncation_side=truncation_side,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        pooled_embs = _pool_last_hidden_state(
            outputs.last_hidden_state,
            inputs.get("attention_mask"),
            pooling=pooling,
        ).cpu().numpy()
        all_embs.append(pooled_embs)

    stacked = np.vstack(all_embs)
    return _normalise_embedding_array(stacked) if normalize else stacked


def build_pair_features(
    query_embs: Tensor,
    chunk_embs: Tensor,
) -> Tensor:
    """Build interaction features for query-context scoring.

    The plain concatenation ``[query, chunk]`` tells the MLP what each vector
    is, but not how they differ.  Adding ``|query-chunk|`` and ``query*chunk``
    exposes mismatch and alignment information directly, which is often useful
    for ranking tasks built on frozen embeddings.

    Args:
        query_embs: Tensor of shape ``(..., hidden_size)``.
        chunk_embs: Tensor of shape ``(..., hidden_size)``.

    Returns:
        Tensor of shape ``(..., hidden_size * 4)`` containing
        ``[query, chunk, abs(query - chunk), query * chunk]``.
    """
    abs_diff = torch.abs(query_embs - chunk_embs)
    product = query_embs * chunk_embs
    return torch.cat([query_embs, chunk_embs, abs_diff, product], dim=-1)


def get_pair_feature_dim(embedding_dim: int) -> int:
    """Return the scorer input width for the interaction feature builder."""
    return embedding_dim * 4


# ---------------------------------------------------------------------------
# MLP Scorer
# ---------------------------------------------------------------------------

class HCCSScorer(nn.Module):
    """MLP scorer that predicts context quality for hallucination prevention.

    Input:  interaction features built from a query embedding and a context
            embedding: ``[query, context, |query-context|, query*context]``.
            Default: 768 * 4 = 3072 dimensions.
    Output: scalar score in [0, 1].  Higher = better hallucination-prevention
            potential for this (query, context) pair.

    Architecture:
        Linear(input_dim, hidden_dim) → ReLU → Dropout(dropout)
        → Linear(hidden_dim, 1) → Sigmoid

    ~1.57 M trainable parameters with defaults
    (3072*512 + 512 + 512*1 + 1).

    Args:
        input_dim:   Pair feature size.  Default 3072 (768 * 4).
        hidden_dim:  Hidden layer width.  Default 512.
        dropout:     Dropout probability.  Default 0.1.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_logits(self, x: Tensor) -> Tensor:
        """Compute quality scores for a batch of (query, context) pairs.

        Args:
            x: Tensor of shape ``(batch_size, input_dim)`` — pair features for
               query and context embeddings.

        Returns:
            Tensor of shape ``(batch_size, 1)`` with unbounded logits.
        """
        return self.net(x)

    def forward(self, x: Tensor) -> Tensor:
        """Compute sigmoid-normalised scores for a batch of pairs."""
        return torch.sigmoid(self.forward_logits(x))

    def score_chunks(
        self,
        query_emb: np.ndarray,
        chunk_embs: np.ndarray,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Score all chunks for a single query; return scores as numpy array.

        Convenience method for inference.  Runs in no-grad mode.

        Args:
            query_emb:  Shape ``(hidden_size,)``.
            chunk_embs: Shape ``(n_chunks, hidden_size)``.
            device:     Torch device string.  Inferred if None.

        Returns:
            1-D numpy array of shape ``(n_chunks,)`` with scores in ``[0, 1]``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        n_chunks = chunk_embs.shape[0]
        query_tensor = torch.tensor(query_emb, dtype=torch.float32, device=device)
        chunk_tensor = torch.tensor(chunk_embs, dtype=torch.float32, device=device)
        query_repeated = query_tensor.unsqueeze(0).expand(n_chunks, -1)
        pair_features = build_pair_features(query_repeated, chunk_tensor)

        x = pair_features
        self.to(device)

        with torch.no_grad():
            scores = self.forward(x)  # (n_chunks, 1)

        return scores.squeeze(1).cpu().numpy()

    def save(self, path: Path) -> None:
        """Save model state dict to disk.

        Args:
            path: Destination ``.pt`` file path.
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: Path,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> "HCCSScorer":
        """Load a saved scorer from a state dict file.

        Args:
            path:       Path to the ``.pt`` file produced by ``save()``.
            input_dim:  Optional scorer input width.  Inferred from the saved
                        state dict when omitted.
            hidden_dim: Optional hidden width.  Inferred from the saved state
                        dict when omitted.
            dropout:    Dropout value (not stored in state dict; used for init).

        Returns:
            Loaded ``HCCSScorer`` instance in eval mode.
        """
        state_dict = torch.load(path, map_location="cpu")
        inferred_hidden_dim = int(state_dict["net.0.weight"].shape[0])
        inferred_input_dim = int(state_dict["net.0.weight"].shape[1])

        scorer = cls(
            input_dim=input_dim or inferred_input_dim,
            hidden_dim=hidden_dim or inferred_hidden_dim,
            dropout=dropout,
        )
        scorer.load_state_dict(state_dict)
        scorer.eval()
        return scorer


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def infonce_loss(
    pos_score: Tensor,
    neg_score: Tensor,
    tau: float = 0.07,
) -> Tensor:
    """InfoNCE contrastive loss for a batch of (positive, negative) pairs.

    Encourages ``pos_score >> neg_score``.  Uses cross-entropy over a
    two-class softmax (positive vs. negative) scaled by temperature ``tau``.

    A lower temperature makes the model more confident — small differences in
    scores become large differences in probability.

    Args:
        pos_score: Scorer output for positive (helpful) context.
                   Shape ``(batch,)`` or ``(batch, 1)``.
        neg_score: Scorer output for negative (unhelpful) context.
                   Shape ``(batch,)`` or ``(batch, 1)``.
        tau:       Temperature parameter.  Default 0.07.

    Returns:
        Scalar loss tensor.
    """
    pos_score = pos_score.view(-1)
    neg_score = neg_score.view(-1)

    # Stack into [pos/tau, neg/tau] for each sample → shape (B, 2)
    logits = torch.stack([pos_score / tau, neg_score / tau], dim=1)
    # Target: class 0 (positive) is always correct
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return nn.functional.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Training loop (stub — implemented in notebooks/02_train_hccs.ipynb)
# ---------------------------------------------------------------------------

def train_hccs(
    scorer: HCCSScorer,
    triplets: List[Any],        # list[ContrastiveTriplet] from data_pipeline
    tokenizer: Any,
    encoder: Any,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> List[float]:
    """Train the HCCS scorer using contrastive triplets.

    The encoder is kept frozen throughout.  Only ``scorer`` parameters receive
    gradient updates.

    Args:
        scorer:         ``HCCSScorer`` instance (on the correct device).
        triplets:       List of ``ContrastiveTriplet`` dataclass instances.
        tokenizer:      HuggingFace tokenizer for the frozen encoder.
        encoder:        Frozen HuggingFace encoder in eval mode.
        epochs:         Number of training epochs.  Default 10.
        lr:             Learning rate for AdamW optimizer.  Default 1e-3.
        device:         Torch device string.  Inferred if None.
        checkpoint_dir: Save ``hccs_epoch_N.pt`` after each epoch when provided.

    Returns:
        List of per-epoch mean training losses.

    TODO:
        1. Build DataLoader from triplets (batch_size=32)
        2. For each batch: call batch_embed on query, pos_ctx, neg_ctx strings
        3. Concat [query_emb, ctx_emb] → scorer.forward → infonce_loss
        4. optimizer.zero_grad() / loss.backward() / optimizer.step()
        5. Log loss; save checkpoint if checkpoint_dir is set
    """
    raise NotImplementedError(
        "train_hccs is implemented in notebooks/02_train_hccs.ipynb. "
        "See that notebook for the full training loop."
    )
