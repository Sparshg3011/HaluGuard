"""
models.py — All HCCS scorer architectures for hallucination-prevention context ranking.

This module contains 7 architectures:
  1. DualEncoder          — single-layer projection + cosine similarity with learned temperature
  2. DualEncoderDeep      — two-layer projection + cosine similarity with learned temperature
  3. ListwiseMLP          — concat [q, c] → MLP logit (listwise cross-entropy loss)
  4. PairwiseMLP          — concat [q, c] → MLP logit (ReLU stack; returns raw logits)
  5. InteractionMLP       — ESIM features [q,c,|q-c|,q*c] → spectral-norm MLP
  6. BilinearScorer       — factorized bilinear q^T W c (fewest parameters)
  7. EnsembleScorer       — learned softmax mixture over individual model logits

All models expose a unified ``score(query_emb, chunk_embs) -> Tensor`` interface
that returns per-chunk **logits** (not probabilities) of shape ``(n_chunks,)`` and
a ``forward(query_embs, chunk_embs)`` that returns batched logits ``(B, n_chunks)``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Shared helpers ------------------------------------------------------------

def _build_interaction_features(query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
    """Concatenate ESIM-style interaction features ``[q, c, |q-c|, q*c]``.

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
    """L2-normalise a tensor along ``dim`` with numerical-stability epsilon."""
    return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))


# 1. DualEncoder ------------------------------------------------------------

class DualEncoder(nn.Module):
    """Single-layer dual encoder with cosine similarity scoring.

    Separate linear projections for query and chunk, followed by LayerNorm
    and GELU activation.  Scoring uses scaled dot-product (cosine similarity)
    with a learnable temperature parameter initialised at log(1/0.07) ≈ 2.66.
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
        self.logit_scale = nn.Parameter(torch.tensor(2.66))

    def _encode_query(self, query_emb: Tensor) -> Tensor:
        return _l2_normalize(self.q_proj(query_emb))

    def _encode_chunks(self, chunk_embs: Tensor) -> Tensor:
        return _l2_normalize(self.c_proj(chunk_embs))

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks for a single query; returns ``(n_chunks,)`` logits."""
        q = self._encode_query(query_emb.unsqueeze(0)).squeeze(0)
        c = self._encode_chunks(chunk_embs)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.mv(c, q) * scale

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        q = _l2_normalize(self.q_proj(query_embs))
        c = _l2_normalize(self.c_proj(chunk_embs))
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.einsum("bd,bnd->bn", q, c) * scale

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, emb_dim: int = 768, proj_dim: int = 128, dropout: float = 0.3) -> "DualEncoder":
        model = cls(emb_dim=emb_dim, proj_dim=proj_dim, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()


# 2. DualEncoderDeep --------------------------------------------------------

class DualEncoderDeep(nn.Module):
    """Two-layer dual encoder with cosine similarity scoring."""

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


# 3. ListwiseMLP ------------------------------------------------------------

class ListwiseMLP(nn.Module):
    """MLP scorer trained with listwise cross-entropy loss.

    Concatenates ``[query_emb, chunk_emb]`` and passes through a single hidden
    layer.  Returns raw logits for use with ``F.cross_entropy``.
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
        _, n_chunks, _ = chunk_embs.shape
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


# 4. PairwiseMLP ------------------------------------------------------------

class PairwiseMLP(nn.Module):
    """MLP scorer with concat + ReLU stack, returning raw logits.

    Concatenates ``[query_emb, chunk_emb]`` and uses a ReLU hidden stack.
    Returns **unbounded logits** (no sigmoid) so it is compatible with the
    listwise cross-entropy path used by ``training.train_listwise_epoch``.

    For an InfoNCE/binary training framing, apply ``torch.sigmoid`` at the
    call site — never bake it into the forward pass.
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
        )

    def forward_pair(self, query_emb: Tensor, chunk_emb: Tensor) -> Tensor:
        """Score a single (query, chunk) pair; returns a scalar logit."""
        x = torch.cat([query_emb, chunk_emb], dim=-1)
        return self.net(x)

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks for a single query; returns ``(n_chunks,)`` logits."""
        n = chunk_embs.shape[0]
        q = query_emb.unsqueeze(0).expand(n, -1)
        x = torch.cat([q, chunk_embs], dim=-1)
        return self.net(x).squeeze(-1)

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        _, n_chunks, _ = chunk_embs.shape
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


# 5. InteractionMLP ---------------------------------------------------------

class InteractionMLP(nn.Module):
    """ESIM-style interaction MLP with spectral normalisation.

    Uses rich interaction features ``[q, c, |q-c|, q*c]`` (4 × emb_dim) fed
    into a two-hidden-layer MLP.  All linear layers use spectral
    normalisation to constrain the Lipschitz constant — one of the most
    effective regularisers for small discriminative nets on frozen embeddings.
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
        _, n_chunks, _ = chunk_embs.shape
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


# 6. BilinearScorer ---------------------------------------------------------

class BilinearScorer(nn.Module):
    """Factorised bilinear scorer: ``score = q_proj · c_proj / sqrt(rank)``.

    Lowest-capacity model in the collection.  The full bilinear form
    ``q^T W c`` (768×768 = 590 K parameters) is factorised as two projections
    into a low-rank space of dimension ``rank`` (~98 K parameters at rank=64).
    Unlike :class:`DualEncoder`, there is no L2 normalisation before scoring,
    so the magnitude of projections carries information.
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
        q = self.q_proj(self.emb_drop(query_emb))
        c = self.c_proj(self.emb_drop(chunk_embs))
        return torch.mv(c, q) / self.scale

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        q = self.q_proj(self.emb_drop(query_embs))
        c = self.c_proj(self.emb_drop(chunk_embs))
        return torch.einsum("br,bnr->bn", q, c) / self.scale

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, emb_dim: int = 768, rank: int = 64, dropout: float = 0.15) -> "BilinearScorer":
        model = cls(emb_dim=emb_dim, rank=rank, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()


# 7. EnsembleScorer ---------------------------------------------------------

class EnsembleScorer(nn.Module):
    """Learned softmax mixture over pre-trained individual scorer logits.

    Each base scorer contributes one logit vector per example; a learnable
    parameter vector of length ``n_scorers``, passed through softmax, gives
    the mixture weights.  The combined logit is ``sum_i w_i * logit_i``.

    The base scorers can be frozen (default) so only the mixture weights
    are optimised in a lightweight second-stage training step.
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
        # Learnable mixture weights — initialised uniform.
        self.gate = nn.Parameter(torch.full((n,), 1.0 / n))

        if freeze_base:
            for scorer in self.scorers:
                for p in scorer.parameters():
                    p.requires_grad_(False)

    def _mixture_weights(self) -> Tensor:
        return F.softmax(self.gate, dim=0)

    def _gather_logits(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Run all base scorers and stack logits: ``(n_scorers, n_chunks)``."""
        parts: List[Tensor] = []
        for scorer in self.scorers:
            with torch.no_grad():
                logits = scorer.score(query_emb, chunk_embs)
            parts.append(logits)
        return torch.stack(parts, dim=0)

    def score(self, query_emb: Tensor, chunk_embs: Tensor) -> Tensor:
        """Score all chunks; returns ``(n_chunks,)`` mixture logits."""
        stacked = self._gather_logits(query_emb, chunk_embs)
        weights = self._mixture_weights()
        return torch.mv(stacked.T, weights)

    def forward(self, query_embs: Tensor, chunk_embs: Tensor) -> Tensor:
        """Batch forward — returns ``(B, n_chunks)`` logits."""
        combined: List[Tensor] = []
        for b in range(query_embs.shape[0]):
            combined.append(self.score(query_embs[b], chunk_embs[b]))
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


# Registry ------------------------------------------------------------------

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

    Raises:
        KeyError: If *name* is not in :data:`MODEL_REGISTRY`.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choose from: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
