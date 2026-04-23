"""
eval_matrix.py — Cached generation/retrieval evaluation helpers.

The functions here power ``notebooks/05_cached_generation_evaluation.ipynb``.
They operate on already-cached tensors and checkpoint files; they never create
embeddings or train models.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from haluguard.artifacts import (
    cceval_method_skip_reason,
)
from haluguard.baselines import (
    bm25_scores,
    cosine_scores,
    edit_similarity_scores,
    full_context_select,
    gold_only_select,
    jaccard_scores,
    random_ranking,
)
from haluguard.benchmarks.base import BenchmarkLoader, Example
from haluguard.evaluate import ALL_METRICS, compute_metrics
from haluguard.generate import build_completion_prompt
from haluguard.models import EnsembleScorer, build_model
from haluguard.retrieval_benchmark import (
    build_ranking_result,
    summarise_rankings,
)
from haluguard.type_router import (
    AstSymbolRouter,
    NoOpTypeRouter,
    RegexTypeRouter,
    TypeRouterBase,
    apply_router_boosts,
)


BASELINE_METHODS: Sequence[str] = (
    "no_context",
    "random_top5",
    "bm25",
    "jaccard",
    "edit_similarity",
    "full_context",
)

REPOBENCH_COSINE_METHODS: Sequence[Tuple[str, str, str]] = (
    ("cosine_codebert_last3", "codebert", "last3"),
    ("cosine_codebert_full", "codebert", "full"),
    ("cosine_unixcoder_last3", "unixcoder", "last3"),
    ("cosine_unixcoder_full", "unixcoder", "full"),
)

SCORER_METHODS: Sequence[str] = (
    "dual_encoder",
    "dual_encoder_deep",
    "listwise_mlp",
    "pairwise_mlp",
    "interaction_mlp",
    "bilinear",
    "ensemble",
)

ROUTER_NAMES: Sequence[str] = ("noop", "regex", "ast_symbol")
DEFAULT_SCORER_BACKEND = "unixcoder"
DEFAULT_SCORER_QUERY_VIEW = "last3"
DEFAULT_METRICS: Sequence[str] = ALL_METRICS


@dataclass
class MethodPlan:
    """One method row in the cached evaluation matrix."""

    benchmark: str
    method: str
    family: str
    status: str = "ready"
    skip_reason: Optional[str] = None
    backend: Optional[str] = None
    query_view: Optional[str] = None
    scorer_name: Optional[str] = None
    router_name: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        """Return a JSON/table friendly row."""
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Summary for one completed or skipped method."""

    benchmark: str
    method: str
    family: str
    status: str
    n: int = 0
    elapsed_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    skip_reason: Optional[str] = None
    details_path: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        """Flatten summary metrics for tables."""
        row = {
            "benchmark": self.benchmark,
            "method": self.method,
            "family": self.family,
            "status": self.status,
            "n": self.n,
            "elapsed_seconds": self.elapsed_seconds,
            "skip_reason": self.skip_reason,
            "details_path": self.details_path,
        }
        row.update(self.metrics)
        return row


def build_router(name: str) -> TypeRouterBase:
    """Construct a router by matrix name."""
    if name == "noop":
        return NoOpTypeRouter()
    if name == "regex":
        return RegexTypeRouter()
    if name == "ast_symbol":
        return AstSymbolRouter()
    raise KeyError(f"Unknown router: {name}")


def _checkpoint_exists(checkpoint_dirs: Sequence[Path], name: str) -> Optional[Path]:
    for directory in checkpoint_dirs:
        path = Path(directory) / f"{name}_best.pt"
        if path.exists():
            return path
    return None


def checkpoint_status(checkpoint_dirs: Sequence[Path]) -> Dict[str, Optional[Path]]:
    """Return checkpoint path for each scorer, or ``None`` when missing."""
    return {
        name: _checkpoint_exists(checkpoint_dirs, name)
        for name in SCORER_METHODS
    }


def build_method_plan(
    benchmark: str,
    data_dir: Path,
    checkpoint_dirs: Sequence[Path],
) -> List[MethodPlan]:
    """Build all method rows, marking unavailable rows as skipped."""
    plans: List[MethodPlan] = [
        MethodPlan(benchmark=benchmark, method=name, family="baseline")
        for name in BASELINE_METHODS
    ]

    if benchmark == "repobench":
        plans.append(MethodPlan(benchmark=benchmark, method="gold_only", family="oracle"))
        for method, backend, query_view in REPOBENCH_COSINE_METHODS:
            plans.append(
                MethodPlan(
                    benchmark=benchmark,
                    method=method,
                    family="cosine",
                    backend=backend,
                    query_view=query_view,
                )
            )
    else:
        plans.extend(
            [
                MethodPlan(
                    benchmark=benchmark,
                    method=method,
                    family="cosine",
                    status="skipped",
                    skip_reason="skipped_missing_cached_embeddings",
                    backend=backend,
                    query_view=query_view,
                )
                for method, backend, query_view in REPOBENCH_COSINE_METHODS
            ]
        )
        for plan in plans:
            if plan.family != "cosine":
                continue
            reason = cceval_method_skip_reason(data_dir, plan.backend, plan.query_view)
            if reason is None:
                plan.status = "ready"
                plan.skip_reason = None
            else:
                plan.status = "skipped"
                plan.skip_reason = reason

    ckpts = checkpoint_status(checkpoint_dirs)
    for scorer_name in SCORER_METHODS:
        ckpt = ckpts.get(scorer_name)
        for router_name in ROUTER_NAMES:
            method = f"{scorer_name}__{router_name}"
            status = "ready"
            reason: Optional[str] = None
            if ckpt is None:
                status = "skipped"
                reason = "skipped_missing_checkpoint"
            elif benchmark != "repobench":
                reason = cceval_method_skip_reason(
                    data_dir,
                    DEFAULT_SCORER_BACKEND,
                    DEFAULT_SCORER_QUERY_VIEW,
                )
                if reason is not None:
                    status = "skipped"
            plans.append(
                MethodPlan(
                    benchmark=benchmark,
                    method=method,
                    family="haluguard",
                    status=status,
                    skip_reason=reason,
                    backend=DEFAULT_SCORER_BACKEND,
                    query_view=DEFAULT_SCORER_QUERY_VIEW,
                    scorer_name=scorer_name,
                    router_name=router_name,
                )
            )

    return plans


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _selected_snippets(example: Example, indices: Sequence[int]) -> List[str]:
    return [example.context_chunks[int(i)]["snippet"] for i in indices]


def _valid_indices(example: Example, indices: Sequence[int]) -> List[int]:
    """Keep only selected indices that exist in the current example."""
    n_chunks = len(example.context_chunks)
    return [int(i) for i in indices if 0 <= int(i) < n_chunks]


def _rank_scores(scores: Sequence[float], top_k: int) -> List[int]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape[0] == 0:
        return []
    k = min(int(top_k), int(arr.shape[0]))
    return np.argsort(arr)[::-1][:k].tolist()


def _complete_ranking(example: Example, ranked_indices: Sequence[int]) -> List[int]:
    """Return a valid exhaustive ranking over all candidates for retrieval."""
    n_chunks = len(example.context_chunks)
    seen = set()
    ordered: List[int] = []
    for idx in ranked_indices:
        value = int(idx)
        if 0 <= value < n_chunks and value not in seen:
            ordered.append(value)
            seen.add(value)
    for idx in range(n_chunks):
        if idx not in seen:
            ordered.append(idx)
    return ordered


def _cceval_artifact_skip_reason(
    plan: MethodPlan,
    cceval_artifacts: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return a skip reason when loaded CrossCodeEval cache mismatches a plan."""
    if cceval_artifacts is None:
        return "skipped_missing_cached_embeddings"
    meta = cceval_artifacts.get("meta") or {}
    backend = meta.get("backend")
    query_view = meta.get("query_view")
    if backend is None or query_view is None:
        return "skipped_incompatible_cached_embeddings"
    if str(backend) != str(plan.backend) or str(query_view) != str(plan.query_view):
        return "skipped_incompatible_cached_embeddings"
    return None


def select_baseline(
    method: str,
    example: Example,
    top_k: int,
    seed: int = 0,
) -> List[int]:
    """Select context indices for a text-only baseline method."""
    n_chunks = len(example.context_chunks)
    if method == "no_context":
        return []
    if method == "random_top5":
        return random_ranking(n_chunks, seed=seed + int(example.metadata.get("source_index", 0)))[:top_k]
    if method == "bm25":
        return _rank_scores(bm25_scores(example.cropped_code, example.context_chunks), top_k)
    if method == "jaccard":
        return _rank_scores(jaccard_scores(example.cropped_code, example.context_chunks), top_k)
    if method == "edit_similarity":
        return _rank_scores(edit_similarity_scores(example.cropped_code, example.context_chunks), top_k)
    if method == "full_context":
        return full_context_select(n_chunks)
    if method == "gold_only":
        if example.gold_index is None:
            return []
        return gold_only_select(example.gold_index)
    raise KeyError(f"Unknown baseline method: {method}")


def select_cosine(
    example_index: int,
    query_embeddings: Any,
    chunk_embeddings: Any,
    top_k: int,
    max_candidates: Optional[int] = None,
) -> List[int]:
    """Select top-k by cosine similarity using cached embeddings."""
    query_emb = _to_numpy(query_embeddings[int(example_index)])
    chunk_embs = _to_numpy(chunk_embeddings[int(example_index)])
    if max_candidates is not None:
        chunk_embs = chunk_embs[:int(max_candidates)]
    return _rank_scores(cosine_scores(query_emb, chunk_embs), top_k)


def _load_scorer(
    scorer_name: str,
    checkpoint_dirs: Sequence[Path],
    device: str,
) -> Optional[Any]:
    """Load one scorer checkpoint, returning ``None`` if unavailable."""
    import torch

    ckpt = _checkpoint_exists(checkpoint_dirs, scorer_name)
    if ckpt is None:
        return None

    if scorer_name == "ensemble":
        member_names = [
            name for name in ("interaction_mlp", "dual_encoder_deep", "dual_encoder")
            if _checkpoint_exists(checkpoint_dirs, name) is not None
        ]
        members = [
            _load_scorer(name, checkpoint_dirs, device)
            for name in member_names
        ]
        members = [m for m in members if m is not None]
        if len(members) < 2:
            return None
        scorer = EnsembleScorer(members)
        scorer.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    else:
        scorer = build_model(scorer_name)
        scorer.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

    scorer.to(device)
    scorer.eval()
    return scorer


def select_haluguard(
    example_index: int,
    example: Example,
    scorer: Any,
    router: TypeRouterBase,
    query_embeddings: Any,
    chunk_embeddings: Any,
    top_k: int,
    device: str = "cpu",
) -> List[int]:
    """Select top-k with a loaded HaluGuard scorer and router."""
    import torch

    query_emb = _to_numpy(query_embeddings[int(example_index)])
    chunk_embs = _to_numpy(chunk_embeddings[int(example_index)])
    chunk_embs = chunk_embs[:len(example.context_chunks)]
    if chunk_embs.shape[0] == 0:
        return []
    contexts = example.context_chunks[:int(chunk_embs.shape[0])]

    q = torch.as_tensor(query_emb, dtype=torch.float32, device=device)
    c = torch.as_tensor(chunk_embs, dtype=torch.float32, device=device)
    with torch.no_grad():
        raw_scores = scorer.score(q, c).detach().cpu().numpy()
    adjusted = apply_router_boosts(raw_scores, contexts, router, example.cropped_code)
    return _rank_scores(adjusted, top_k)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str))
            handle.write("\n")


def run_generation_method(
    plan: MethodPlan,
    loader: BenchmarkLoader,
    generator: Callable[[List[str]], List[str]],
    output_dir: Path,
    repobench_artifacts: Optional[Dict[str, Any]] = None,
    cceval_artifacts: Optional[Dict[str, Any]] = None,
    checkpoint_dirs: Optional[Sequence[Path]] = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    limit: int = 200,
    batch_size: int = 16,
    top_k: int = 5,
    device: str = "cpu",
    encoder: Any = None,
    tokenizer: Any = None,
) -> EvaluationSummary:
    """Run one method row and return a summary.

    Skipped methods are returned immediately and written into the aggregate
    table by the caller.
    """
    if plan.status != "ready":
        return EvaluationSummary(
            benchmark=plan.benchmark,
            method=plan.method,
            family=plan.family,
            status=plan.status,
            skip_reason=plan.skip_reason,
        )

    start = time.time()
    checkpoint_dirs = list(checkpoint_dirs or [])
    records: List[Dict[str, Any]] = []
    prompts: List[str] = []
    references: List[str] = []
    source_indices: List[int] = []
    selected_indices: List[List[int]] = []

    scorer = None
    router: Optional[TypeRouterBase] = None
    if plan.family == "haluguard":
        assert plan.scorer_name is not None
        assert plan.router_name is not None
        scorer = _load_scorer(plan.scorer_name, checkpoint_dirs, device)
        if scorer is None:
            return EvaluationSummary(
                benchmark=plan.benchmark,
                method=plan.method,
                family=plan.family,
                status="skipped",
                skip_reason="skipped_missing_checkpoint",
            )
        router = build_router(plan.router_name)

    if plan.benchmark == "repobench":
        if repobench_artifacts is None:
            raise ValueError("RepoBench methods require repobench_artifacts")
        query_embeddings = repobench_artifacts["query_embeddings"]
        chunk_embeddings = repobench_artifacts["chunk_embeddings"]
    else:
        query_embeddings = cceval_artifacts["query_embeddings"] if cceval_artifacts else None
        chunk_embeddings = cceval_artifacts["chunk_embeddings"] if cceval_artifacts else None

    for example in loader.iter_examples(limit=limit):
        source_idx = int(example.metadata.get("source_index", len(source_indices)))
        selected: List[int]

        if plan.family in ("baseline", "oracle"):
            selected = select_baseline(plan.method, example, top_k=top_k)
        elif plan.family == "cosine":
            if plan.benchmark != "repobench":
                reason = _cceval_artifact_skip_reason(plan, cceval_artifacts)
                if reason is not None:
                    return EvaluationSummary(
                        benchmark=plan.benchmark,
                        method=plan.method,
                        family=plan.family,
                        status="skipped",
                        skip_reason=reason,
                    )
            if query_embeddings is None or chunk_embeddings is None:
                return EvaluationSummary(
                    benchmark=plan.benchmark,
                    method=plan.method,
                    family=plan.family,
                    status="skipped",
                    skip_reason="skipped_missing_cached_embeddings",
                )
            selected = select_cosine(
                source_idx,
                query_embeddings[(plan.backend, plan.query_view)] if plan.benchmark == "repobench" else query_embeddings,
                chunk_embeddings[plan.backend] if plan.benchmark == "repobench" else chunk_embeddings,
                top_k=top_k,
                max_candidates=len(example.context_chunks),
            )
        elif plan.family == "haluguard":
            if plan.benchmark != "repobench":
                reason = _cceval_artifact_skip_reason(plan, cceval_artifacts)
                if reason is not None:
                    return EvaluationSummary(
                        benchmark=plan.benchmark,
                        method=plan.method,
                        family=plan.family,
                        status="skipped",
                        skip_reason=reason,
                    )
            if query_embeddings is None or chunk_embeddings is None or scorer is None or router is None:
                return EvaluationSummary(
                    benchmark=plan.benchmark,
                    method=plan.method,
                    family=plan.family,
                    status="skipped",
                    skip_reason="skipped_missing_cached_embeddings",
                )
            selected = select_haluguard(
                source_idx,
                example,
                scorer,
                router,
                query_embeddings[(plan.backend, plan.query_view)] if plan.benchmark == "repobench" else query_embeddings,
                chunk_embeddings[plan.backend] if plan.benchmark == "repobench" else chunk_embeddings,
                top_k=top_k,
                device=device,
            )
        else:
            raise KeyError(f"Unknown method family: {plan.family}")

        selected = _valid_indices(example, selected)
        prompt = build_completion_prompt(
            cropped_code=example.cropped_code,
            import_statement=example.import_statement,
            selected_snippets=_selected_snippets(example, selected),
        )
        prompts.append(prompt)
        references.append(example.reference)
        source_indices.append(source_idx)
        selected_indices.append(selected)

    predictions: List[str] = []
    for start_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        predictions.extend(generator(batch_prompts))

    for source_idx, selected, pred, ref in zip(source_indices, selected_indices, predictions, references):
        records.append(
            {
                "source_index": source_idx,
                "prediction": pred,
                "reference": ref,
                "selected_indices": selected,
                "method": plan.method,
                "benchmark": plan.benchmark,
            }
        )

    details_path = Path(output_dir) / f"{plan.benchmark}__{plan.method}.jsonl"
    _write_jsonl(details_path, records)
    metric_values = compute_metrics(
        predictions,
        references,
        metrics=metrics,
        encoder=encoder,
        tokenizer=tokenizer,
        device=device,
    )
    return EvaluationSummary(
        benchmark=plan.benchmark,
        method=plan.method,
        family=plan.family,
        status="completed",
        n=len(predictions),
        elapsed_seconds=round(time.time() - start, 2),
        metrics=metric_values,
        details_path=str(details_path),
    )


def compute_repobench_retrieval_table(
    plans: Sequence[MethodPlan],
    loader: BenchmarkLoader,
    repobench_artifacts: Dict[str, Any],
    checkpoint_dirs: Sequence[Path],
    limit: int = 200,
    top_k: int = 5,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Compute RepoBench retrieval metrics for all ready retrieval methods."""
    rankings = []
    examples = list(loader.iter_examples(limit=limit))

    scorer_cache: Dict[str, Any] = {}
    for plan in plans:
        if plan.status != "ready":
            continue
        if plan.family not in ("baseline", "cosine", "haluguard"):
            continue
        if plan.method in ("no_context", "full_context"):
            continue

        if plan.family == "haluguard" and plan.scorer_name not in scorer_cache:
            scorer_cache[plan.scorer_name or ""] = _load_scorer(
                plan.scorer_name or "",
                checkpoint_dirs,
                device,
            )

        for example in examples:
            if example.gold_index is None:
                continue
            src_idx = int(example.metadata.get("source_index", 0))
            if plan.family == "baseline":
                ranked = select_baseline(plan.method, example, top_k=len(example.context_chunks))
            elif plan.family == "cosine":
                ranked = select_cosine(
                    src_idx,
                    repobench_artifacts["query_embeddings"][(plan.backend, plan.query_view)],
                    repobench_artifacts["chunk_embeddings"][plan.backend],
                    top_k=len(example.context_chunks),
                    max_candidates=len(example.context_chunks),
                )
            else:
                scorer = scorer_cache.get(plan.scorer_name or "")
                if scorer is None:
                    continue
                ranked = select_haluguard(
                    src_idx,
                    example,
                    scorer,
                    build_router(plan.router_name or "noop"),
                    repobench_artifacts["query_embeddings"][(plan.backend, plan.query_view)],
                    repobench_artifacts["chunk_embeddings"][plan.backend],
                    top_k=len(example.context_chunks),
                    device=device,
                )
            rankings.append(
                build_ranking_result(
                    method=plan.method,
                    example_id=src_idx,
                    query_view=plan.query_view or "",
                    candidate_count=len(example.context_chunks),
                    gold_index=int(example.gold_index),
                    ranked_indices=_complete_ranking(example, ranked),
                )
            )

    return summarise_rankings(rankings, include_mrr=True)


def build_router_delta_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute router-vs-noop metric deltas for HaluGuard rows."""
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if "__" not in method:
            continue
        scorer, router = method.rsplit("__", 1)
        by_key[(str(row.get("benchmark")), f"{scorer}__{router}")] = row

    deltas: List[Dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        if not method.endswith("__noop"):
            continue
        benchmark = str(row.get("benchmark"))
        scorer = method[:-len("__noop")]
        base_em = row.get("em")
        for router_name in ("regex", "ast_symbol"):
            other = by_key.get((benchmark, f"{scorer}__{router_name}"))
            if other is None or other.get("status") != "completed":
                continue
            other_em = other.get("em")
            delta = None
            if isinstance(base_em, (int, float)) and isinstance(other_em, (int, float)):
                delta = float(other_em) - float(base_em)
            deltas.append(
                {
                    "benchmark": benchmark,
                    "scorer": scorer,
                    "router": router_name,
                    "noop_em": base_em,
                    "router_em": other_em,
                    "delta_em": delta,
                }
            )
    return deltas


def write_table_files(rows: Sequence[Dict[str, Any]], base_path: Path) -> None:
    """Write table rows to JSON, CSV, and Markdown using one base path."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = base_path.with_suffix(".json")
    csv_path = base_path.with_suffix(".csv")
    md_path = base_path.with_suffix(".md")

    rows_list = list(rows)
    json_path.write_text(json.dumps(rows_list, indent=2, default=str), encoding="utf-8")

    headers: List[str] = []
    for row in rows_list:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)

    def _fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows_list:
        lines.append("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fake_generator(prompts: List[str]) -> List[str]:
    """Small deterministic generator for smoke tests and local notebook checks."""
    out: List[str] = []
    for prompt in prompts:
        lines = [line for line in prompt.splitlines() if line.strip()]
        out.append(lines[-1].strip() if lines else "")
    return out
