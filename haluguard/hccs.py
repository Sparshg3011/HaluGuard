"""
hccs.py — Hallucination Context Contrast Scorer (HCCS).

Architecture:
    Frozen CodeBERT encoder (microsoft/codebert-base, 768-dim CLS token)
    + concat(query_emb, context_emb) → 1536-dim input
    → Linear(1536, 256) → ReLU → Dropout(0.1) → Linear(256, 1) → Sigmoid

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
from typing import Any, List, Optional

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

def embed_code(
    text: str,
    tokenizer: Any,
    model: Any,
    device: Optional[str] = None,
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

    Returns:
        1-D numpy array of shape ``(hidden_size,)``, e.g. ``(768,)`` for
        ``microsoft/codebert-base``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer.truncation_side = 'left'

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (batch=1, seq_len, hidden_size)
    # Index 0 → CLS token, the sentence-level representation
    cls_emb: Tensor = outputs.last_hidden_state[:, 0, :]
    cls_emb = F.normalize(cls_emb, p=2, dim=0)
    return cls_emb.squeeze(0).cpu().numpy()


def batch_embed(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    device: Optional[str] = None,
    batch_size: int = 32,
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

    Returns:
        2-D numpy array of shape ``(len(texts), hidden_size)``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_embs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls_embs)

    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# MLP Scorer
# ---------------------------------------------------------------------------

class HCCSScorer(nn.Module):
    """MLP scorer that predicts context quality for hallucination prevention.

    Input:  concatenation of a query embedding and a context embedding.
            Default: 768 + 768 = 1536 dimensions.
    Output: scalar score in [0, 1].  Higher = better hallucination-prevention
            potential for this (query, context) pair.

    Architecture:
        Linear(input_dim, hidden_dim) → ReLU → Dropout(dropout)
        → Linear(hidden_dim, 1) → Sigmoid

    ~394 K trainable parameters with defaults (1536*256 + 256 + 256*1 + 1).

    Args:
        input_dim:   Concatenated embedding size.  Default 1536 (768 * 2).
        hidden_dim:  Hidden layer width.  Default 256.
        dropout:     Dropout probability.  Default 0.1.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # bounds output to (-1, 1) without killing gradients like Sigmoid
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute quality scores for a batch of (query, context) pairs.

        Args:
            x: Tensor of shape ``(batch_size, input_dim)`` — concatenated
               query and context embeddings.

        Returns:
            Tensor of shape ``(batch_size, 1)`` with scores in ``[0, 1]``.
        """
        return self.net(x)

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
        # Repeat query embedding to match chunk count
        query_repeated = np.tile(query_emb, (n_chunks, 1))  # (n_chunks, 768)
        combined = np.concatenate([query_repeated, chunk_embs], axis=1)  # (n_chunks, 1536)

        x = torch.tensor(combined, dtype=torch.float32).to(device)
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
        input_dim: int = 1536,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> "HCCSScorer":
        """Load a saved scorer from a state dict file.

        Args:
            path:       Path to the ``.pt`` file produced by ``save()``.
            input_dim:  Must match the saved model's architecture.
            hidden_dim: Must match the saved model's architecture.
            dropout:    Dropout value (not stored in state dict; used for init).

        Returns:
            Loaded ``HCCSScorer`` instance in eval mode.
        """
        scorer = cls(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        scorer.load_state_dict(torch.load(path, map_location="cpu"))
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
    # with raw logits, division by tau directly controls margin sharpness
    logits = torch.stack([pos_score / tau, neg_score / tau], dim=1)
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
