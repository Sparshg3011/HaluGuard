"""
pipeline.py — End-to-end HaluGuard inference pipeline for RepoBench.

Wires together all components for inference::

    cropped_code + context chunks
        → CodeBERT (embed query + chunks)
        → Scorer (rank chunks by prevention score)
        → TypeRouter (pre-emptive boost based on code patterns)
        → EFL (generate → execute → boost scores → retry)
        → predicted next line

The pipeline works with **any** of the 7 scorer architectures declared in
:mod:`haluguard.models` (``DualEncoder``, ``DualEncoderDeep``, ``ListwiseMLP``,
``PairwiseMLP``, ``InteractionMLP``, ``BilinearScorer``, ``EnsembleScorer``)
as well as the legacy :class:`HCCSScorer`.  The scorer must expose either
``score_chunks(query_emb, chunk_embs, device=...)`` (legacy) or
``score(query_emb, chunk_embs)`` returning a torch Tensor.

Usage::

    pipeline = HaluGuardPipeline.from_checkpoint("checkpoints/hccs_best.pt")
    result = pipeline.run(
        cropped_code=example["cropped_code"],
        import_statement=example["import_statement"],
        contexts=example["context"],
        generate_fn=my_llm_call,
    )
    print(result["prediction"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from haluguard.efl import EFLResult, run_efl
from haluguard.hccs import HCCSScorer, batch_embed, embed_code
from haluguard.type_router import boost_scores, predict_boost


class HaluGuardPipeline:
    """End-to-end pipeline that wraps all HaluGuard components.

    Attributes:
        scorer:    Trained scorer (any of the 7 architectures or legacy HCCS).
        tokenizer: HuggingFace tokenizer for the encoder.
        encoder:   Frozen HuggingFace encoder in eval mode.
        top_k:     Number of top-scoring chunks to include in the initial prompt.
        device:    Torch device string.
        verbose:   If True, print step-by-step progress.
        scorer_name: Short human-readable label for the scorer (used in prints).
    """

    def __init__(
        self,
        scorer: Any,
        tokenizer: Any,
        encoder: Any,
        top_k: int = 5,
        device: Optional[str] = None,
        verbose: bool = False,
        scorer_name: Optional[str] = None,
    ) -> None:
        """Initialise the pipeline with pre-loaded components.

        Prefer :meth:`from_checkpoint` or :meth:`from_trained` for most cases.
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scorer = scorer.to(device)
        self.scorer.eval()
        self.tokenizer = tokenizer
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.top_k = top_k
        self.device = device
        self.verbose = verbose
        self.scorer_name = scorer_name or type(scorer).__name__

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        top_k: int = 5,
        encoder_name: str = "microsoft/codebert-base",
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> "HaluGuardPipeline":
        """Load a pipeline from a saved (legacy) HCCS scorer checkpoint."""
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        encoder = AutoModel.from_pretrained(encoder_name)
        encoder.eval()

        scorer = HCCSScorer.load(Path(checkpoint_path))

        return cls(
            scorer=scorer,
            tokenizer=tokenizer,
            encoder=encoder,
            top_k=top_k,
            device=device,
            verbose=verbose,
            scorer_name="HCCSScorer",
        )

    @classmethod
    def from_trained(
        cls,
        scorer: Any,
        tokenizer: Any,
        encoder: Any,
        top_k: int = 5,
        device: Optional[str] = None,
        verbose: bool = False,
        scorer_name: Optional[str] = None,
    ) -> "HaluGuardPipeline":
        """Wrap an already-instantiated scorer with a shared encoder/tokenizer.

        Preferred path when benchmarking multiple architectures, since it
        avoids re-downloading/instantiating the encoder for every model.
        """
        return cls(
            scorer=scorer,
            tokenizer=tokenizer,
            encoder=encoder,
            top_k=top_k,
            device=device,
            verbose=verbose,
            scorer_name=scorer_name,
        )

    # Unified scoring ------------------------------------------------------

    def _score(
        self,
        query_emb: np.ndarray,
        chunk_embs: np.ndarray,
    ) -> np.ndarray:
        """Score all chunks against a query with whichever scorer is loaded.

        Returns per-chunk scores as a 1-D numpy array of length ``n_chunks``.
        Dispatches to :meth:`HCCSScorer.score_chunks` for the legacy model
        and to :meth:`score` for any architecture in :mod:`haluguard.models`.
        """
        if chunk_embs.shape[0] == 0:
            return np.array([], dtype=np.float32)

        if isinstance(self.scorer, HCCSScorer):
            return self.scorer.score_chunks(query_emb, chunk_embs, device=self.device)

        import torch

        q = torch.as_tensor(query_emb, dtype=torch.float32, device=self.device)
        c = torch.as_tensor(chunk_embs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.scorer.score(q, c)
        return logits.detach().cpu().numpy().astype(np.float32)

    def select_contexts(
        self,
        query_emb: np.ndarray,
        chunk_embs: np.ndarray,
        contexts: List[Dict[str, str]],
        cropped_code: str,
    ) -> List[int]:
        """Score and rank context chunks, applying pre-emptive type-router boosts."""
        if not contexts:
            return []

        scores = self._score(query_emb, chunk_embs)
        boosts = predict_boost(cropped_code)
        adjusted_scores = boost_scores(scores, contexts, boosts)

        k = min(self.top_k, len(contexts))
        sorted_indices = np.argsort(adjusted_scores)[::-1][:k]
        if self.verbose:
            print(f"[pipeline:{self.scorer_name}] selected top-{k} of {len(contexts)} chunks")
        return sorted_indices.tolist()

    def run(
        self,
        cropped_code: str,
        import_statement: str,
        contexts: List[Dict[str, str]],
        generate_fn: Callable[[str], str],
        max_iterations: int = 3,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """Run the full HaluGuard pipeline for one RepoBench example."""
        if self.verbose:
            print(
                f"[pipeline:{self.scorer_name}] run — contexts={len(contexts)}, "
                f"top_k={self.top_k}, max_iterations={max_iterations}"
            )

        snippets = [c["snippet"] for c in contexts]
        query_emb = embed_code(
            cropped_code, self.tokenizer, self.encoder, device=self.device
        )
        if snippets:
            chunk_embs = batch_embed(
                snippets, self.tokenizer, self.encoder, device=self.device
            )
        else:
            chunk_embs = np.empty((0, query_emb.shape[0]), dtype=np.float32)

        scores = self._score(query_emb, chunk_embs)
        boosts = predict_boost(cropped_code)
        adjusted_scores = (
            boost_scores(scores, contexts, boosts) if len(scores) > 0 else scores
        )
        selected_indices = self.select_contexts(
            query_emb, chunk_embs, contexts, cropped_code
        )

        efl_result: EFLResult = run_efl(
            cropped_code=cropped_code,
            import_statement=import_statement,
            contexts=contexts,
            scores=adjusted_scores,
            generate_fn=generate_fn,
            top_k=self.top_k,
            max_iterations=max_iterations,
            timeout=timeout,
            verbose=self.verbose,
        )

        if self.verbose:
            print(
                f"[pipeline:{self.scorer_name}] done — passed={efl_result.passed}, "
                f"iterations={efl_result.iterations}"
            )

        return {
            "prediction": efl_result.code,
            "passed": efl_result.passed,
            "iterations": efl_result.iterations,
            "history": efl_result.history,
            "selected_indices": selected_indices,
        }
