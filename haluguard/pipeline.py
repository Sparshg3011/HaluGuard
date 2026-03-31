"""
pipeline.py — End-to-end HaluGuard inference pipeline for RepoBench.

Wires together all components for inference:
    cropped_code + context chunks
        → CodeBERT (embed query + chunks)
        → HCCSScorer (rank chunks by prevention score)
        → TypeRouter (pre-emptive boost based on code patterns)
        → EFL (generate → execute → boost scores → retry)
        → predicted next line

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
        scorer:    Trained ``HCCSScorer`` used to rank context chunks.
        tokenizer: HuggingFace tokenizer for the CodeBERT encoder.
        encoder:   Frozen CodeBERT encoder in eval mode.
        top_k:     Number of top-scoring chunks to include in the initial prompt.
        device:    Torch device string.
    """

    def __init__(
        self,
        scorer: HCCSScorer,
        tokenizer: Any,
        encoder: Any,
        top_k: int = 5,
        device: Optional[str] = None,
    ) -> None:
        """Initialise the pipeline with pre-loaded components.

        Prefer ``HaluGuardPipeline.from_checkpoint`` for the common case of
        loading a saved scorer.

        Args:
            scorer:    Trained and loaded ``HCCSScorer``.
            tokenizer: HuggingFace tokenizer (e.g. for ``microsoft/codebert-base``).
            encoder:   Frozen HuggingFace encoder in eval mode.
            top_k:     How many highest-scoring chunks to include.  Default 5.
            device:    ``"cuda"`` or ``"cpu"``.  Inferred if None.
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        top_k: int = 5,
        encoder_name: str = "microsoft/codebert-base",
        device: Optional[str] = None,
    ) -> "HaluGuardPipeline":
        """Load a pipeline from a saved HCCS scorer checkpoint.

        Downloads CodeBERT from HuggingFace on first call (cached thereafter).

        Args:
            checkpoint_path: Path to the ``.pt`` file saved by ``HCCSScorer.save()``.
            top_k:           Number of top chunks to use.  Default 5.
            encoder_name:    HuggingFace model ID for the encoder.
            device:          Torch device string.  Inferred if None.

        Returns:
            Initialised ``HaluGuardPipeline``.
        """
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
        )

    def select_contexts(
        self,
        query_emb: np.ndarray,
        chunk_embs: np.ndarray,
        contexts: List[Dict[str, str]],
        cropped_code: str,
    ) -> List[int]:
        """Score and rank context chunks, applying pre-emptive type-router boosts.

        Args:
            query_emb:   Shape ``(hidden_size,)`` — query embedding.
            chunk_embs:  Shape ``(n_chunks, hidden_size)`` — chunk embeddings.
            contexts:    List of context dicts with ``"snippet"`` and ``"path"``.
            cropped_code: The code written so far (for pre-emptive analysis).

        Returns:
            List of up to ``self.top_k`` chunk indices, sorted by descending
            boosted score.
        """
        if not contexts:
            return []

        # HCCS scoring
        scores = self.scorer.score_chunks(
            query_emb, chunk_embs, device=self.device
        )

        # Pre-emptive type-router boosting
        boosts = predict_boost(cropped_code)
        adjusted_scores = boost_scores(scores, contexts, boosts)

        # Select top-k
        k = min(self.top_k, len(contexts))
        sorted_indices = np.argsort(adjusted_scores)[::-1][:k]
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
        """Run the full HaluGuard pipeline for one RepoBench example.

        Steps:
          1. Embed the query (cropped_code) and all context snippets.
          2. Score and select top-k chunks using HCCS + type-router boosts.
          3. Run the Execution Feedback Loop with score-based re-ranking.

        Args:
            cropped_code:    Code written so far in the current file.
            import_statement: Import statements from the current file.
            contexts:        List of context dicts, each with ``"snippet"``,
                             ``"path"``, and ``"identifier"`` keys.
            generate_fn:     Callable(prompt: str) → predicted next line.
            max_iterations:  Max EFL retries.  Default 3.
            timeout:         Subprocess timeout per execution attempt.

        Returns:
            Dict with keys:
                ``prediction`` (str), ``passed`` (bool), ``iterations`` (int),
                ``history`` (list of ExecutionResult),
                ``selected_indices`` (list of int).
        """
        # Step 1: Embed query and chunks
        snippets = [c["snippet"] for c in contexts]
        query_emb = embed_code(
            cropped_code, self.tokenizer, self.encoder, device=self.device
        )
        chunk_embs = batch_embed(
            snippets, self.tokenizer, self.encoder, device=self.device
        ) if snippets else np.empty((0, query_emb.shape[0]))

        # Step 2: HCCS scoring + pre-emptive boost
        scores = self.scorer.score_chunks(
            query_emb, chunk_embs, device=self.device
        ) if len(chunk_embs) > 0 else np.array([])

        boosts = predict_boost(cropped_code)
        adjusted_scores = boost_scores(scores, contexts, boosts) if len(scores) > 0 else scores

        selected_indices = self.select_contexts(
            query_emb, chunk_embs, contexts, cropped_code
        )

        # Step 3: EFL with score-based re-ranking
        efl_result: EFLResult = run_efl(
            cropped_code=cropped_code,
            import_statement=import_statement,
            contexts=contexts,
            scores=adjusted_scores,
            generate_fn=generate_fn,
            top_k=self.top_k,
            max_iterations=max_iterations,
            timeout=timeout,
        )

        return {
            "prediction": efl_result.code,
            "passed": efl_result.passed,
            "iterations": efl_result.iterations,
            "history": efl_result.history,
            "selected_indices": selected_indices,
        }
