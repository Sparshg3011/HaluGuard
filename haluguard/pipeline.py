"""
pipeline.py — End-to-end HaluGuard inference pipeline.

Wires together all components for inference:
    repo_files + query
        → chunker (split into chunks)
        → CodeBERT (embed query + chunks)
        → HCCSScorer (rank chunks by prevention score)
        → TypeRouter (add error-type-targeted context)
        → EFL (generate → execute → retry)
        → result

Usage::

    pipeline = HaluGuardPipeline.from_checkpoint("checkpoints/hccs_best.pt")
    result = pipeline.run(
        query="Write get_forecast using WeatherReport...",
        repo_files={"api.py": "...", "models.py": "..."},
        test_code="assert isinstance(get_forecast('London'), WeatherReport)",
        generate_fn=my_llm_call,
    )
    print(result["passed"], result["code"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from haluguard.chunker import chunk_repo
from haluguard.efl import EFLResult, run_efl
from haluguard.hccs import HCCSScorer, batch_embed, embed_code
from haluguard.type_router import route


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

        TODO:
            Implement loading once train_hccs produces a checkpoint.
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

    def select_chunks(
        self,
        query: str,
        chunks: List[str],
    ) -> List[str]:
        """Score all chunks for a query and return the top-k highest scorers.

        Args:
            query:  Natural-language coding task.
            chunks: List of repo chunks from ``chunk_repo``.

        Returns:
            Up to ``self.top_k`` chunks, ordered by descending HCCS score.
        """
        if not chunks:
            return []

        query_emb = embed_code(query, self.tokenizer, self.encoder, device=self.device)
        chunk_embs = batch_embed(chunks, self.tokenizer, self.encoder, device=self.device)

        scores = self.scorer.score_chunks(query_emb, chunk_embs, device=self.device)

        # Sort by descending score, take top-k
        sorted_indices = np.argsort(scores)[::-1][: self.top_k]
        return [chunks[i] for i in sorted_indices]

    def run(
        self,
        query: str,
        repo_files: Dict[str, str],
        test_code: str,
        generate_fn: Callable[[str], str],
        max_iterations: int = 3,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Run the full HaluGuard pipeline for one coding task.

        Steps:
          1. Chunk the repo files.
          2. Score and select top-k chunks using the HCCS scorer.
          3. Run the Execution Feedback Loop with targeted context retrieval.

        Args:
            query:          Natural-language coding task.
            repo_files:     Mapping of filepath → source code.
            test_code:      Test assertions to verify generated code.
            generate_fn:    Callable(prompt: str) → generated code string.
            max_iterations: Max EFL retries.  Default 3.
            timeout:        Subprocess timeout per execution attempt.

        Returns:
            Dict with keys:
                ``code`` (str), ``passed`` (bool), ``iterations`` (int),
                ``history`` (list of ExecutionResult), ``selected_chunks`` (list of str).
        """
        # Step 1: chunk repo
        chunks = chunk_repo(repo_files)

        # Step 2: HCCS scoring — select top-k chunks
        selected_chunks = self.select_chunks(query, chunks)

        # Step 3: EFL with targeted fallback retrieval
        efl_result: EFLResult = run_efl(
            query=query,
            initial_context=selected_chunks,
            test_code=test_code,
            generate_fn=generate_fn,
            repo_files=repo_files,
            max_iterations=max_iterations,
            timeout=timeout,
        )

        return {
            "code": efl_result.code,
            "passed": efl_result.passed,
            "iterations": efl_result.iterations,
            "history": efl_result.history,
            "selected_chunks": selected_chunks,
        }
