"""
evaluate.py — Metrics computation for hallucination evaluation.

All metric functions are pure Python (no ML imports required).  This makes
them fast to run and easy to test independently.

The evaluation runners (which call the full pipeline on benchmark datasets)
are implemented in notebooks/03_evaluation.ipynb.

Metrics defined here:
    - Hallucination Rate (HR): fraction of samples with at least one hallucination
    - Per-type HR: HR broken down by hallucination category
    - Hallucination Reduction Ratio: relative improvement over a baseline
    - Pass Rate: fraction of samples where generated code passes all tests
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_hallucination_rate(results: List[Dict[str, Any]]) -> float:
    """Compute the overall Hallucination Rate (HR).

    HR = number of samples with a hallucination / total samples.

    Each result dict must have a boolean key ``"hallucinated"`` (True if the
    generated code contained a hallucination of any type).

    Args:
        results: List of per-sample result dicts.  Each must have
                 ``"hallucinated": bool``.

    Returns:
        HR as a float in ``[0.0, 1.0]``.  Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0

    n_hallucinated = sum(1 for r in results if r.get("hallucinated", False))
    return n_hallucinated / len(results)


def compute_per_type_hr(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute Hallucination Rate broken down by hallucination type.

    Args:
        results: List of per-sample result dicts.  Each may have
                 ``"hallucination_type": str`` (e.g. ``"resource"``) and
                 ``"hallucinated": bool``.  Samples without a type are
                 grouped under ``"unknown"``.

    Returns:
        Dict mapping hallucination type → HR for that type.
        E.g. ``{"resource": 0.32, "naming": 0.18, "mapping": 0.11, "logic": 0.05}``.
    """
    from collections import defaultdict

    type_total: Dict[str, int] = defaultdict(int)
    type_hallucinated: Dict[str, int] = defaultdict(int)

    for r in results:
        hall_type = r.get("hallucination_type") or "unknown"
        type_total[hall_type] += 1
        if r.get("hallucinated", False):
            type_hallucinated[hall_type] += 1

    return {
        t: type_hallucinated[t] / type_total[t]
        for t in type_total
    }


def compute_reduction_ratio(baseline_hr: float, method_hr: float) -> float:
    """Compute the Hallucination Reduction Ratio relative to a baseline.

    Reduction Ratio = (HR_baseline - HR_method) / HR_baseline

    A ratio of 0.5 means the method cut hallucinations in half.  A negative
    ratio means the method is worse than the baseline.

    Args:
        baseline_hr: HR of the baseline method (e.g. BM25 retrieval).
        method_hr:   HR of the method under evaluation (e.g. HaluGuard).

    Returns:
        Reduction ratio as a float.  Returns 0.0 when baseline_hr is 0.

    Raises:
        ValueError: If either HR is outside ``[0.0, 1.0]``.
    """
    if not (0.0 <= baseline_hr <= 1.0):
        raise ValueError(f"baseline_hr must be in [0, 1], got {baseline_hr}")
    if not (0.0 <= method_hr <= 1.0):
        raise ValueError(f"method_hr must be in [0, 1], got {method_hr}")

    if baseline_hr == 0.0:
        return 0.0

    return (baseline_hr - method_hr) / baseline_hr


def compute_pass_rate(results: List[Dict[str, Any]]) -> float:
    """Compute the fraction of samples where generated code passed all tests.

    Pass Rate = passed_samples / total_samples.

    Args:
        results: List of per-sample result dicts.  Each must have
                 ``"passed": bool``.

    Returns:
        Pass rate as a float in ``[0.0, 1.0]``.  Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0

    n_passed = sum(1 for r in results if r.get("passed", False))
    return n_passed / len(results)


def compute_metrics_table(
    results_by_method: Dict[str, List[Dict[str, Any]]],
    baseline_key: str = "no_context",
) -> List[Dict[str, Any]]:
    """Compute a summary metrics table across multiple methods.

    Args:
        results_by_method: Dict mapping method name → list of per-sample results.
        baseline_key:      Key in ``results_by_method`` to use as the baseline
                           for computing reduction ratios.  Default ``"no_context"``.

    Returns:
        List of dicts, one per method, with keys:
        ``method``, ``hr``, ``pass_rate``, ``reduction_ratio``.
        Sorted by ascending HR.
    """
    baseline_results = results_by_method.get(baseline_key, [])
    baseline_hr = compute_hallucination_rate(baseline_results)

    rows: List[Dict[str, Any]] = []
    for method, results in results_by_method.items():
        hr = compute_hallucination_rate(results)
        rows.append({
            "method": method,
            "hr": hr,
            "pass_rate": compute_pass_rate(results),
            "reduction_ratio": compute_reduction_ratio(baseline_hr, hr),
        })

    return sorted(rows, key=lambda r: r["hr"])


# ---------------------------------------------------------------------------
# Evaluation runners (stubs — implemented in notebooks/03_evaluation.ipynb)
# ---------------------------------------------------------------------------

def evaluate_baseline(
    method: str,
    tasks: List[Dict[str, Any]],
    generate_fn: Any,
    retrieval_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run a baseline method on a list of tasks and return per-sample results.

    Each task dict must contain:
        - ``query``      (str): natural-language task description
        - ``repo_files`` (dict): filepath → source mapping
        - ``test_code``  (str): test assertions
        - ``task_id``    (str): unique identifier

    Args:
        method:       One of ``"no_context"``, ``"bm25"``, ``"full_context"``,
                      ``"codebert_similarity"``.
        tasks:        List of task dicts from CodeHaluEval.
        generate_fn:  Callable ``(prompt: str) -> str`` returning generated code.
        retrieval_fn: For ``"bm25"`` and ``"codebert_similarity"``, a callable
                      ``(query, chunks) -> List[str]`` that selects top-k chunks.
                      If None, full context is used as fallback.

    Returns:
        List of per-sample result dicts with keys:
        ``task_id``, ``method``, ``passed``, ``hallucinated``,
        ``hallucination_type``, ``error_type``.
    """
    from haluguard.chunker import chunk_repo
    from haluguard.efl import execute_code

    results: List[Dict[str, Any]] = []

    for task in tasks:
        task_id = task.get("task_id", "")
        query = task.get("query", "")
        repo_files = task.get("repo_files", {})
        test_code = task.get("test_code", "")

        chunks = chunk_repo(repo_files)

        if method == "no_context":
            selected_chunks: List[str] = []
        elif method == "full_context":
            selected_chunks = chunks
        elif retrieval_fn is not None:
            selected_chunks = retrieval_fn(query, chunks)
        else:
            selected_chunks = chunks  # fallback for unknown method

        context_str = "\n\n".join(selected_chunks) if selected_chunks else "(no context)"
        prompt = (
            "# Repository context:\n\n"
            + context_str
            + "\n\n# Task:\n# "
            + query
            + "\n\n# Write the implementation below:\n"
        )

        code = generate_fn(prompt)
        exec_result = execute_code(code, test_code)

        results.append({
            "task_id": task_id,
            "method": method,
            "passed": exec_result.passed,
            "hallucinated": not exec_result.passed,
            "hallucination_type": (
                exec_result.hallucination_type.value
                if exec_result.hallucination_type else None
            ),
            "error_type": exec_result.error_type,
        })

    return results


def evaluate_haluguard(
    tasks: List[Dict[str, Any]],
    pipeline: Any,  # HaluGuardPipeline from pipeline.py
    generate_fn: Any = None,
) -> List[Dict[str, Any]]:
    """Run the full HaluGuard pipeline on a list of tasks.

    Each task dict must contain the same keys as ``evaluate_baseline``.

    Args:
        tasks:        List of task dicts from CodeHaluEval.
        pipeline:     Initialised ``HaluGuardPipeline`` instance.
        generate_fn:  Callable ``(prompt: str) -> str``.  If None, falls back
                      to ``pipeline.generate_fn`` if that attribute exists.

    Returns:
        List of per-sample result dicts (same schema as ``evaluate_baseline``),
        plus an ``"iterations"`` key showing how many EFL retries were used.
    """
    results: List[Dict[str, Any]] = []

    for task in tasks:
        task_id = task.get("task_id", "")
        query = task.get("query", "")
        repo_files = task.get("repo_files", {})
        test_code = task.get("test_code", "")

        _generate_fn = generate_fn or getattr(pipeline, "generate_fn", None)
        if _generate_fn is None:
            raise ValueError(
                "generate_fn must be provided either as an argument or as "
                "pipeline.generate_fn"
            )

        run_result = pipeline.run(
            query=query,
            repo_files=repo_files,
            test_code=test_code,
            generate_fn=_generate_fn,
        )

        last_history = run_result["history"]
        hall_type = None
        error_type = None
        if last_history and not run_result["passed"]:
            last = last_history[-1]
            hall_type = last.hallucination_type.value if last.hallucination_type else None
            error_type = last.error_type

        results.append({
            "task_id": task_id,
            "method": "haluguard",
            "passed": run_result["passed"],
            "hallucinated": not run_result["passed"],
            "hallucination_type": hall_type,
            "error_type": error_type,
            "iterations": run_result["iterations"],
        })

    return results
