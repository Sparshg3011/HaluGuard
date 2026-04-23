"""
run_eval.py — Unified, crash-safe benchmark runner.

Runs a single (benchmark × scorer × router) cell end-to-end:
  1. Walk the benchmark loader, collecting per-example selected context.
  2. Batch-generate predictions via ``generate_next_line_batch``.
  3. (Optionally) run EFL per example to re-rank + retry.
  4. Compute the requested metrics on the predictions.
  5. Write per-example JSONL + a final per-cell JSON summary.

Design constraints:
  - Every per-example call is wrapped in ``try/except``; one bad example
    cannot abort the full evaluation run.
  - JSONL is appended **as the cell runs** so a crash loses at most one
    example.
  - Re-running with the same ``cell_id`` / ``resume_dir`` skips indices that
    are already present in the JSONL.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from haluguard.benchmarks.base import BenchmarkLoader, Example
from haluguard.efl import EFLResult, run_efl
from haluguard.evaluate import (
    ALL_METRICS,
    compute_metrics,
    first_error_type_hist,
)
from haluguard.generate import build_completion_prompt, generate_next_line_batch
from haluguard.hccs import batch_embed, embed_code
from haluguard.pipeline import HaluGuardPipeline
from haluguard.type_router import TypeRouterBase, apply_router_boosts

# Type alias: source_index → (query_emb shape (768,), chunk_embs shape (n_chunks, 768))
PrecomputedEmbeddings = Dict[int, Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_existing_indices(jsonl_path: Path) -> Dict[int, Dict[str, Any]]:
    """Return ``{source_index: record}`` already written to ``jsonl_path``."""
    if not jsonl_path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("source_index")
            if isinstance(idx, int):
                out[idx] = rec
    return out


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON record + newline, flushing immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str))
        fh.write("\n")
        fh.flush()


def _make_default_generator(
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    max_prompt_tokens: int = 2048,
) -> Callable[[List[str]], List[str]]:
    """Build a batched-generator callable wrapping ``generate_next_line_batch``."""

    def _gen(prompts: List[str]) -> List[str]:
        return generate_next_line_batch(
            prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_prompt_tokens=max_prompt_tokens,
        )

    return _gen


# ---------------------------------------------------------------------------
# Selection & prompt assembly (non-EFL path)
# ---------------------------------------------------------------------------

def _select_and_prompt(
    pipeline: HaluGuardPipeline,
    example: Example,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[str, List[int]]:
    """Select top-k chunks for ``example`` and build the completion prompt.

    Args:
        pipeline:     The fully initialised :class:`HaluGuardPipeline`.
        example:      The benchmark example to process.
        precomputed:  Optional ``(query_emb, chunk_embs)`` tuple.  When
                      provided the CodeBERT forward passes are skipped,
                      giving a ~30× speedup for the full evaluation matrix.

    Returns:
        The prompt string and the list of selected chunk indices.
    """
    cropped = example.cropped_code
    contexts = example.context_chunks

    if precomputed is not None:
        query_emb, chunk_embs = precomputed
    else:
        query_emb = embed_code(
            cropped,
            pipeline.tokenizer,
            pipeline.encoder,
            device=pipeline.device,
        )
        if contexts:
            snippets = [c["snippet"] for c in contexts]
            chunk_embs = batch_embed(
                snippets,
                pipeline.tokenizer,
                pipeline.encoder,
                device=pipeline.device,
            )
        else:
            chunk_embs = np.empty((0, query_emb.shape[0]), dtype=np.float32)

    if contexts:
        selected = pipeline.select_contexts(query_emb, chunk_embs, contexts, cropped)
    else:
        selected = []

    selected_snippets = [contexts[i]["snippet"] for i in selected]
    prompt = build_completion_prompt(
        cropped_code=cropped,
        import_statement=example.import_statement,
        selected_snippets=selected_snippets,
    )
    return prompt, selected


# ---------------------------------------------------------------------------
# EFL path (per-example, slower but honours retries + error-type histogram)
# ---------------------------------------------------------------------------

def _run_efl_for_example(
    pipeline: HaluGuardPipeline,
    example: Example,
    generate_single: Callable[[str], str],
    max_iterations: int,
    timeout: int,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[str, List[int], EFLResult]:
    """Run the full EFL for one example and return (prediction, selected, result).

    Args:
        pipeline:       The fully initialised :class:`HaluGuardPipeline`.
        example:        The benchmark example to process.
        generate_single: Single-prompt generator function.
        max_iterations: EFL retry cap.
        timeout:        Subprocess timeout per EFL execution.
        precomputed:    Optional ``(query_emb, chunk_embs)`` tuple.  When
                        provided the CodeBERT forward passes are skipped.
    """
    cropped = example.cropped_code
    contexts = example.context_chunks

    if precomputed is not None:
        query_emb, chunk_embs = precomputed
    else:
        query_emb = embed_code(
            cropped,
            pipeline.tokenizer,
            pipeline.encoder,
            device=pipeline.device,
        )
        if contexts:
            snippets = [c["snippet"] for c in contexts]
            chunk_embs = batch_embed(
                snippets,
                pipeline.tokenizer,
                pipeline.encoder,
                device=pipeline.device,
            )
        else:
            chunk_embs = np.empty((0, query_emb.shape[0]), dtype=np.float32)

    scores = pipeline._score(query_emb, chunk_embs)
    adjusted_scores = (
        apply_router_boosts(scores, contexts, pipeline.router, cropped)
        if len(scores) > 0
        else scores
    )
    selected = pipeline.select_contexts(query_emb, chunk_embs, contexts, cropped)

    efl_result = run_efl(
        cropped_code=cropped,
        import_statement=example.import_statement,
        contexts=contexts,
        scores=adjusted_scores,
        generate_fn=generate_single,
        top_k=pipeline.top_k,
        max_iterations=max_iterations,
        timeout=timeout,
        verbose=False,
        router=pipeline.router,
    )
    return efl_result.code, selected, efl_result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    loader: BenchmarkLoader,
    pipeline: HaluGuardPipeline,
    generator: Callable[[List[str]], List[str]],
    generate_single: Optional[Callable[[str], str]] = None,
    metrics: Sequence[str] = ALL_METRICS,
    limit: int = 250,
    batch_size: int = 16,
    use_efl: bool = False,
    max_iterations: int = 3,
    timeout: int = 10,
    cell_id: Optional[str] = None,
    resume_dir: Optional[Path] = None,
    progress_every: int = 10,
    codebert_encoder: Any = None,
    codebert_tokenizer: Any = None,
    precomputed_embeddings: Optional[PrecomputedEmbeddings] = None,
) -> Dict[str, Any]:
    """Evaluate ``pipeline`` on ``loader`` for one (scorer, router) cell.

    Args:
        loader:             Benchmark loader (:class:`BenchmarkLoader`).
        pipeline:           Fully initialised :class:`HaluGuardPipeline`.
        generator:          Batched generator — list of prompts → list of
                            predicted next lines. Typically wraps
                            :func:`generate_next_line_batch`.
        generate_single:    Single-prompt generator required when
                            ``use_efl=True``. If omitted, one is synthesised
                            from ``generator``.
        metrics:            Subset of :data:`ALL_METRICS` to compute.
        limit:              Max number of examples to evaluate.
        batch_size:         Mini-batch for the non-EFL (bulk) path.
        use_efl:            If True, run the full EFL per example (slower but
                            exposes an error-type histogram).
        max_iterations:     EFL retry cap.
        timeout:            Subprocess timeout per EFL execution.
        cell_id:            Stable identifier for checkpoints. If ``None``,
                            derived from ``loader.name`` + ``pipeline.scorer_name``
                            + ``pipeline.router.name``.
        resume_dir:         Directory for per-cell ``.jsonl`` + ``.json`` files.
                            Defaults to ``./data/results/cells``.
        progress_every:     Print a progress line every N examples.
        codebert_encoder:        Encoder for the ``codebert_score`` metric (usually
                                 reuses ``pipeline.encoder``).
        codebert_tokenizer:      Tokenizer for ``codebert_score`` (reuses
                                 ``pipeline.tokenizer``).
        precomputed_embeddings:  Optional mapping ``{source_index: (query_emb,
                                 chunk_embs)}`` of pre-computed CodeBERT
                                 embeddings.  When an entry is present for an
                                 example the embedding step is skipped entirely,
                                 giving a ~30× speedup for the full evaluation
                                 matrix.

    Returns:
        Summary dict with keys ``cell_id``, ``benchmark``, ``scorer``,
        ``router``, ``n``, ``n_errors``, ``metrics``, ``error_type_hist``,
        ``elapsed_seconds``.
    """
    if cell_id is None:
        cell_id = f"{loader.name}__{pipeline.scorer_name}__{pipeline.router.name}"
    if resume_dir is None:
        resume_dir = Path("data/results/cells")
    resume_dir = Path(resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = resume_dir / f"{cell_id}.jsonl"
    summary_path = resume_dir / f"{cell_id}.json"

    existing = _read_existing_indices(jsonl_path)
    if existing:
        print(
            f"[run_eval] cell={cell_id} — resuming: {len(existing)} existing records"
        )

    if codebert_encoder is None:
        codebert_encoder = pipeline.encoder
    if codebert_tokenizer is None:
        codebert_tokenizer = pipeline.tokenizer

    # Build single-prompt generator lazily if needed.
    if use_efl and generate_single is None:
        def generate_single(prompt: str) -> str:  # type: ignore[no-redef]
            out = generator([prompt])
            return out[0] if out else ""

    start = time.time()
    predictions: List[str] = []
    references: List[str] = []
    history_entries: List[Dict[str, Any]] = []
    n_errors = 0
    processed = 0

    # ------------------------------------------------------------------
    # Non-EFL fast path: collect all prompts, then batch-generate.
    # ------------------------------------------------------------------
    if not use_efl:
        buffered: List[Tuple[Example, int, str, List[int]]] = []

        def _flush() -> None:
            nonlocal n_errors, processed
            if not buffered:
                return
            prompts = [item[2] for item in buffered]
            try:
                preds = generator(prompts)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[run_eval] batch generate failed: {exc}")
                preds = [""] * len(prompts)
                n_errors += len(prompts)

            for (ex, src_idx, _prompt, selected), pred in zip(buffered, preds):
                record = {
                    "source_index": src_idx,
                    "task_id": ex.metadata.get("task_id"),
                    "prediction": pred,
                    "reference": ex.reference,
                    "selected_indices": selected,
                    "gold_index": ex.gold_index,
                    "history": [],
                }
                _append_jsonl(jsonl_path, record)
                predictions.append(pred)
                references.append(ex.reference)
                history_entries.append(record)
                processed += 1
                if processed % progress_every == 0:
                    print(
                        f"[run_eval] cell={cell_id} processed={processed}/{limit} "
                        f"errors={n_errors}"
                    )
            buffered.clear()

        for example in loader.iter_examples(limit=limit):
            src_idx = example.metadata.get("source_index")
            if src_idx in existing:
                rec = existing[src_idx]
                predictions.append(rec.get("prediction", ""))
                references.append(rec.get("reference", example.reference))
                history_entries.append(rec)
                processed += 1
                continue
            try:
                precomp = (
                    precomputed_embeddings.get(src_idx)
                    if precomputed_embeddings is not None
                    else None
                )
                prompt, selected = _select_and_prompt(pipeline, example, precomputed=precomp)
            except Exception as exc:
                tb = traceback.format_exc(limit=2)
                print(f"[run_eval] example {src_idx} select failed: {exc}")
                _append_jsonl(
                    jsonl_path,
                    {
                        "source_index": src_idx,
                        "task_id": example.metadata.get("task_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "trace": tb,
                    },
                )
                n_errors += 1
                continue
            buffered.append((example, src_idx, prompt, selected))
            if len(buffered) >= batch_size:
                _flush()

        _flush()

    # ------------------------------------------------------------------
    # EFL path: per-example, honours retries and records history.
    # ------------------------------------------------------------------
    else:
        assert generate_single is not None
        for example in loader.iter_examples(limit=limit):
            src_idx = example.metadata.get("source_index")
            if src_idx in existing:
                rec = existing[src_idx]
                predictions.append(rec.get("prediction", ""))
                references.append(rec.get("reference", example.reference))
                history_entries.append(rec)
                processed += 1
                continue
            try:
                precomp = (
                    precomputed_embeddings.get(src_idx)
                    if precomputed_embeddings is not None
                    else None
                )
                pred, selected, efl_result = _run_efl_for_example(
                    pipeline,
                    example,
                    generate_single,
                    max_iterations=max_iterations,
                    timeout=timeout,
                    precomputed=precomp,
                )
                record = {
                    "source_index": src_idx,
                    "task_id": example.metadata.get("task_id"),
                    "prediction": pred,
                    "reference": example.reference,
                    "selected_indices": selected,
                    "gold_index": example.gold_index,
                    "passed": efl_result.passed,
                    "iterations": efl_result.iterations,
                    "history": [
                        {
                            "passed": h.passed,
                            "error_type": h.error_type,
                            "error_message": h.error_message,
                        }
                        for h in efl_result.history
                    ],
                }
            except Exception as exc:
                tb = traceback.format_exc(limit=2)
                print(f"[run_eval] example {src_idx} EFL failed: {exc}")
                record = {
                    "source_index": src_idx,
                    "task_id": example.metadata.get("task_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": tb,
                }
                n_errors += 1

            _append_jsonl(jsonl_path, record)
            if "prediction" in record:
                predictions.append(record["prediction"])
                references.append(record["reference"])
            history_entries.append(record)
            processed += 1
            if processed % progress_every == 0:
                print(
                    f"[run_eval] cell={cell_id} processed={processed}/{limit} "
                    f"errors={n_errors}"
                )

    # ------------------------------------------------------------------
    # Metrics + summary
    # ------------------------------------------------------------------
    metric_dict = compute_metrics(
        predictions,
        references,
        metrics=metrics,
        encoder=codebert_encoder,
        tokenizer=codebert_tokenizer,
        device=pipeline.device,
    )
    err_hist = first_error_type_hist(history_entries) if use_efl else {}

    summary: Dict[str, Any] = {
        "cell_id": cell_id,
        "benchmark": loader.name,
        "scorer": pipeline.scorer_name,
        "router": pipeline.router.name,
        "n": len(predictions),
        "n_errors": n_errors,
        "metrics": metric_dict,
        "error_type_hist": err_hist,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(
        f"[run_eval] cell={cell_id} DONE n={summary['n']} "
        f"errors={n_errors} elapsed={summary['elapsed_seconds']}s"
    )
    return summary


def build_generator_from_model(
    tokenizer: Any,
    model: Any,
    device: str = "cuda",
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    max_prompt_tokens: int = 2048,
) -> Callable[[List[str]], List[str]]:
    """Public helper exposing :func:`_make_default_generator`."""
    return _make_default_generator(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_prompt_tokens=max_prompt_tokens,
    )
