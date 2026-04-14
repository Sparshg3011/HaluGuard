"""
models.py — All HCCS scorer architectures for hallucination-prevention context ranking.

This module contains 7 architectures:
  1. DualEncoder          — single-layer projection + cosine similarity with learned temperature
  2. DualEncoderDeep      — two-layer projection + cosine similarity with learned temperature
  3. ListwiseMLP          — concat [q, c] → MLP logit (listwise cross-entropy loss)
  4. PairwiseMLP          — concat [q, c] → MLP sigmoid (InfoNCE pairwise loss)
  5. InteractionMLP       — ESIM features [q,c,|q-c|,q*c] → spectral-norm MLP (anti-overfit)
  6. BilinearScorer       — factorized bilinear q^T W c (fewest parameters, strong regulariser)
  7. EnsembleScorer       — learned gating over scores from individual models

All models expose a unified `score(query_emb, chunk_embs) -> Tensor` interface that
returns per-chunk logits (not probabilities) of shape (n_chunks,).

Usage example::

    from haluguard.models import DualEncoder, EnsembleScorer
    model = DualEncoder(emb_dim=768, proj_dim=128, dropout=0.3)
    logits = model.score(query_emb, chunk_embs)   # (n_chunks,)
    top_idx = logits.argmax().item()
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_interaction_features(query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
    """Concatenate ESIM-style interaction features: [q, c, |q-c|, q*c].

    Args:
        query_embs: ``(..., emb_dim)``
        chunk_embs: ``(..., emb_dim)``

    Returns:
        Tensor of shape ``(..., emb_dim * 4)``.
    """
    return torch.cat(
        [query_embs, chunk_embs, (query_embs - chunk_embs).abs(), query_embs * chunk_embs],
        dim=-1,
    )


def _l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))


# ---------------------------------------------------------------------------
# 1. DualEncoder
# ---------------------------------------------------------------------------

class DualEncoder(nn.Module):
    """Single-layer dual encoder with cosine similarity scoring.

    Separate linear projections for query and chunk, followed by LayerNorm
    and GELU activation.  Scoring uses scaled dot-product (cosine similarity)
    with a learnable temperature parameter initialised at log(1/0.07) ≈ 2.66.

    Architecture::

        q_proj: Linear(emb_dim, proj_dim) → LayerNorm → GELU → Dropout
        c_proj: Linear(emb_dim, proj_dim) → LayerNorm → GELU → Dropout
        score  = L2_norm(c_proj) · L2_norm(q_proj) * exp(logit_scale)

    Args:
        emb_dim:  Input embedding dimension (e.g. 768 for CodeBERT).
        proj_dim: Projection head output dimension.  Default 128.
        dropout:  Dropout probability applied after GELU.  Default 0.3.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        proj_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.c_proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        # Initialise to 1/0.07 ≈ 14.3, i.e. log scale ≈ 2.66
        self.logit_scale = nn.Parameter(torch.tensor(2.66))

    def _encode_query(self, query_emb: Tensor) -> Tensor:
        return _l2_normalize(self.q_proj(query_emb))

    def _encode_chunks(self, chunk_embs: Tensor) -> Tensor:
        return _l2_normalize(self.c_proj(chunk_embs))

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks for a single query.

        Args:
            query_emb:  Shape ``(emb_dim,)`` — single query vector.
            chunk_embs: Shape ``(n_chunks, emb_dim)``.

        Returns:
            Logits of shape ``(n_chunks,)``.
        """
        q = self._encode_query(query_emb.unsqueeze(0)).squeeze(0)   # (proj_dim,)
        c = self._encode_chunks(chunk_embs)                          # (n_chunks, proj_dim)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.mv(c, q) * scale

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward for training.

        Args:
            query_embs: ``(B, emb_dim)``
            chunk_embs: ``(B, n_chunks, emb_dim)``

        Returns:
            Logits of shape ``(B, n_chunks)``.
        """
        q = _l2_normalize(self.q_proj(query_embs))            # (B, proj_dim)
        c = _l2_normalize(self.c_proj(chunk_embs))            # (B, n_chunks, proj_dim)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.einsum("bd,bnd->bn", q, c) * scale

    def save(self, path: Path) -> None:
        """Save model state dict."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, emb_dim: int = 768, proj_dim: int = 128, dropout: float = 0.3) -> "DualEncoder":
        """Load from checkpoint."""
        model = cls(emb_dim=emb_dim, proj_dim=proj_dim, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()


# ---------------------------------------------------------------------------
# 2. DualEncoderDeep
# ---------------------------------------------------------------------------

class DualEncoderDeep(nn.Module):
    """Two-layer dual encoder with cosine similarity scoring.

    Same as DualEncoder but the projection head has two fully-connected layers,
    giving it more capacity to learn task-specific representations while still
    regularising via LayerNorm and Dropout.

    Architecture::

        q_proj: Linear(emb_dim, 512) → LN → GELU → Dropout
                → Linear(512, proj_dim) → LN
        c_proj: (same structure)
        score  = L2_norm(c_proj) · L2_norm(q_proj) * exp(logit_scale)

    Args:
        emb_dim:  Input embedding dimension.  Default 768.
        proj_dim: Final projection dimension.  Default 128.
        dropout:  Dropout probability.  Default 0.3.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        proj_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden = 512
        self.q_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.c_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.66))

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks for a single query; returns ``(n_chunks,)`` logits."""
        q = _l2_normalize(self.q_proj(query_emb.unsqueeze(0))).squeeze(0)
        c = _l2_normalize(self.c_proj(chunk_embs))
        return torch.mv(c, q) * self.logit_scale.exp().clamp(max=100.0)

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        q = _l2_normalize(self.q_proj(query_embs))
        c = _l2_normalize(self.c_proj(chunk_embs))
        return torch.einsum("bd,bnd->bn", q, c) * self.logit_scale.exp().clamp(max=100.0)

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, emb_dim: int = 768, proj_dim: int = 128, dropout: float = 0.3) -> "DualEncoderDeep":
        model = cls(emb_dim=emb_dim, proj_dim=proj_dim, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()


# ---------------------------------------------------------------------------
# 3. ListwiseMLP
# ---------------------------------------------------------------------------

class ListwiseMLP(nn.Module):
    """MLP scorer trained with listwise cross-entropy loss.

    Concatenates ``[query_emb, chunk_emb]`` (no interaction features) and
    passes through a single hidden layer.  No sigmoid — returns raw logits
    for use with ``F.cross_entropy``.

    Architecture::

        Linear(emb_dim*2, hidden_dim) → LayerNorm → GELU → Dropout → Linear(hidden_dim, 1)

    Args:
        emb_dim:    Input embedding dimension.  Default 768.
        hidden_dim: Hidden layer width.  Default 256.
        dropout:    Dropout probability.  Default 0.3.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks for a single query; returns ``(n_chunks,)`` logits."""
        n = chunk_embs.shape[0]
        q = query_emb.unsqueeze(0).expand(n, -1)
        x = torch.cat([q, chunk_embs], dim=-1)
        return self.net(x).squeeze(-1)

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        B, n_chunks, _ = chunk_embs.shape
        q = query_embs.unsqueeze(1).expand(-1, n_chunks, -1)
        x = torch.cat([q, chunk_embs], dim=-1)
        return self.net(x).squeeze(-1)

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, emb_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.3) -> "ListwiseMLP":
        model = cls(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()


# ---------------------------------------------------------------------------
# 4. PairwiseMLP
# ---------------------------------------------------------------------------

class PairwiseMLP(nn.Module):
    """MLP scorer trained with InfoNCE pairwise loss.

    Concatenates ``[query_emb, chunk_emb]`` and uses ReLU + Sigmoid (binary
    classification framing: does this chunk help?).  Evaluation is done
    listwise (ranking by scores), but training optimises the pairwise objective.

    Architecture::

        Linear(emb_dim*2, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 1) → Sigmoid

    Args:
        emb_dim:    Input embedding dimension.  Default 768.
        hidden_dim: Hidden layer width.  Default 256.
        dropout:    Dropout probability.  Default 0.3.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward_pair(self, query_emb: Tensor, chunk_emb: Tensor) -> Tensor:
        """Score a single (query, chunk) pair; returns scalar in [0, 1]."""
        x = torch.cat([query_emb, chunk_emb], dim=-1)
        return self.net(x)

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks for a single query; returns ``(n_chunks,)`` probs."""
        n = chunk_embs.shape[0]
        q = query_emb.unsqueeze(0).expand(n, -1)
        x = torch.cat([q, chunk_embs], dim=-1)
        return self.net(x).squeeze(-1)

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` probabilities."""
        B, n_chunks, _ = chunk_embs.shape
        q = query_embs.unsqueeze(1).expand(-1, n_chunks, -1)
        x = torch.cat([q, chunk_embs], dim=-1)
        return self.net(x).squeeze(-1)

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, emb_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.3) -> "PairwiseMLP":
        model = cls(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()


# ---------------------------------------------------------------------------
# 5. InteractionMLP  (NEW — spectral norm + deep ESIM features)
# ---------------------------------------------------------------------------

class InteractionMLP(nn.Module):
    """ESIM-style interaction MLP with spectral normalisation.

    Uses rich interaction features ``[q, c, |q-c|, q*c]`` (4 × emb_dim) fed
    into a **two-hidden-layer** MLP.  All linear layers use spectral
    normalisation, which constrains the Lipschitz constant of each layer and
    is one of the most effective anti-overfitting regularisers for small
    discriminative networks trained on frozen embeddings.

    Architecture::

        features = [q, c, |q-c|, q*c]              # (emb_dim*4,)
        SpectralNorm(Linear(emb_dim*4, h1)) → LN → GELU → Dropout
        SpectralNorm(Linear(h1, h2))         → LN → GELU → Dropout
        SpectralNorm(Linear(h2, 1))                          → logit

    Args:
        emb_dim: Input embedding dimension.  Default 768.
        h1:      First hidden layer width.  Default 512.
        h2:      Second hidden layer width.  Default 128.
        dropout: Dropout probability.  Default 0.3.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        h1: int = 512,
        h2: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = emb_dim * 4
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, h1)),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.utils.spectral_norm(nn.Linear(h1, h2)),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.utils.spectral_norm(nn.Linear(h2, 1)),
        )

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks for a single query; returns ``(n_chunks,)`` logits."""
        n = chunk_embs.shape[0]
        q = query_emb.unsqueeze(0).expand(n, -1)
        x = _build_interaction_features(q, chunk_embs)
        return self.net(x).squeeze(-1)

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        B, n_chunks, _ = chunk_embs.shape
        q = query_embs.unsqueeze(1).expand(-1, n_chunks, -1)
        x = _build_interaction_features(q, chunk_embs)
        return self.net(x).squeeze(-1)

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, emb_dim: int = 768, h1: int = 512, h2: int = 128, dropout: float = 0.3) -> "InteractionMLP":
        model = cls(emb_dim=emb_dim, h1=h1, h2=h2, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()


# ---------------------------------------------------------------------------
# 6. BilinearScorer  (NEW — lowest capacity, strongest anti-overfit)
# ---------------------------------------------------------------------------

class BilinearScorer(nn.Module):
    """Factorized bilinear scorer: score = q_proj · c_proj / sqrt(rank).

    This is the lowest-capacity model in the collection.  The full bilinear
    form ``q^T W c`` (768×768 = 590 K parameters) is factorized as two
    projections into a low-rank space of dimension ``rank``.  The result is
    only ``2 × emb_dim × rank`` parameters — e.g. 98 K at rank=64 vs 1.57 M
    for the original HCCSScorer.

    Fewer parameters mean much less overfitting risk while still learning a
    task-specific geometry beyond plain cosine similarity.  Unlike DualEncoder,
    there is no L2 normalisation before scoring, so the magnitude of
    projections carries information too.

    Architecture::

        q_proj: Linear(emb_dim, rank, bias=False)
        c_proj: Linear(emb_dim, rank, bias=False)
        score  = (q_proj(q) · c_proj(c)) / sqrt(rank)

    Args:
        emb_dim: Input embedding dimension.  Default 768.
        rank:    Factorization rank.  Default 64.
        dropout: Input embedding dropout before projection.  Default 0.15.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        rank: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, rank, bias=False)
        self.c_proj = nn.Linear(emb_dim, rank, bias=False)
        self.scale = math.sqrt(rank)
        self.emb_drop = nn.Dropout(p=dropout)

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks for a single query; returns ``(n_chunks,)`` logits."""
        q = self.q_proj(self.emb_drop(query_emb))      # (rank,)
        c = self.c_proj(self.emb_drop(chunk_embs))     # (n_chunks, rank)
        return torch.mv(c, q) / self.scale

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        q = self.q_proj(self.emb_drop(query_embs))     # (B, rank)
        c = self.c_proj(self.emb_drop(chunk_embs))     # (B, n_chunks, rank)
        return torch.einsum("br,bnr->bn", q, c) / self.scale

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, emb_dim: int = 768, rank: int = 64, dropout: float = 0.15) -> "BilinearScorer":
        model = cls(emb_dim=emb_dim, rank=rank, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()


# ---------------------------------------------------------------------------
# 7. EnsembleScorer  (NEW — learned gating over individual model outputs)
# ---------------------------------------------------------------------------

class EnsembleScorer(nn.Module):
    """Learned weighted combination of pre-trained individual scorers.

    Runs each scorer in no-grad mode and learns a small gating network to
    combine their raw logits into a single ranking signal.  Trained in a
    second stage after all individual scorers converge.

    The gating is a single linear layer (no bias) followed by Softmax, so it
    learns which models to trust for which examples.  The combined logit is
    a weighted sum: ``combined = sum_i w_i * logit_i``.

    Args:
        scorers:     Pre-trained scorer instances (any type from this module).
        freeze_base: If True, all base scorers are frozen during training.
                     Default True (only gate is trained).
    """

    def __init__(
        self,
        scorers: Sequence[nn.Module],
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        n = len(scorers)
        assert n >= 2, "EnsembleScorer requires at least 2 base scorers."
        self.scorers = nn.ModuleList(scorers)
        # Gate: (n,) → (n,) weights, no bias — forces it to be a pure mixture
        self.gate = nn.Linear(n, n, bias=False)
        # Initialise to uniform mixture
        nn.init.constant_(self.gate.weight, 1.0 / n)

        if freeze_base:
            for scorer in self.scorers:
                for p in scorer.parameters():
                    p.requires_grad_(False)

    def _gather_logits(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Run all scorers and stack logits: ``(n_scorers, n_chunks)``."""
        parts: List[Tensor] = []
        for scorer in self.scorers:
            with torch.no_grad():
                logits = scorer.score(query_emb, chunk_embs)   # (n_chunks,)
            parts.append(logits)
        return torch.stack(parts, dim=0)   # (n_scorers, n_chunks)

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks; returns ``(n_chunks,)`` combined logits."""
        stacked = self._gather_logits(query_emb, chunk_embs)   # (n_scorers, n_chunks)
        weights = F.softmax(self.gate.weight.sum(dim=1), dim=0)  # (n_scorers,)
        return torch.mv(stacked.T, weights)                      # (n_chunks,)

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        B = query_embs.shape[0]
        combined = []
        for b in range(B):
            logits = self.score(query_embs[b], chunk_embs[b])
            combined.append(logits)
        return torch.stack(combined, dim=0)

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: Path,
        scorers: Sequence[nn.Module],
        freeze_base: bool = True,
    ) -> "EnsembleScorer":
        model = cls(scorers=scorers, freeze_base=freeze_base)
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        return model.eval()


# ---------------------------------------------------------------------------
# Registry — convenient lookup by name
# ---------------------------------------------------------------------------

#: Maps model names to their constructor classes.
MODEL_REGISTRY: Dict[str, type] = {
    "dual_encoder":      DualEncoder,
    "dual_encoder_deep": DualEncoderDeep,
    "listwise_mlp":      ListwiseMLP,
    "pairwise_mlp":      PairwiseMLP,
    "interaction_mlp":   InteractionMLP,
    "bilinear":          BilinearScorer,
    "ensemble":          EnsembleScorer,
}


def build_model(name: str, **kwargs: object) -> nn.Module:
    """Instantiate a scorer by registry name.

    Args:
        name:   One of ``"dual_encoder"``, ``"dual_encoder_deep"``,
                ``"listwise_mlp"``, ``"pairwise_mlp"``, ``"interaction_mlp"``,
                ``"bilinear"``, ``"ensemble"``.
        **kwargs: Constructor keyword arguments forwarded to the model.

    Returns:
        Uninitialised (randomly weighted) model instance.

    Raises:
        KeyError: If *name* is not in ``MODEL_REGISTRY``.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Choose from: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
