"""
efl.py — Execution Feedback Loop (EFL).

Generates the next line of code, optionally executes a validation snippet,
classifies any error, boosts context scores for the failing category, and
retries — up to max_iterations times.

For RepoBench next-line prediction, the EFL constructs a minimal executable
test by combining import statements + cropped code + the predicted line.
If execution fails, context scores are re-ranked using ``error_boost`` and
``boost_scores`` from the type router.

On Colab: No Docker needed.  Colab runs inside an isolated VM.  We use
``subprocess.run`` with a timeout and a temporary file.  The temp file is
always cleaned up in a ``finally`` block.

See docs/DRY_RUN.md for a worked example of the EFL in action.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from haluguard.hccs import HallucinationType
from haluguard.type_router import ERROR_TO_CATEGORY, boost_scores, error_boost


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing LLM-generated code against a test suite.

    Attributes:
        passed:             True if the subprocess exited with code 0.
        stdout:             Captured standard output.
        stderr:             Captured standard error / traceback.
        error_type:         Python exception class name, e.g. ``"ImportError"``.
                            None if the code passed.
        error_message:      First meaningful line of the error message.
                            None if the code passed.
        hallucination_type: Classified hallucination category.
                            None if the code passed.
    """

    passed: bool
    stdout: str
    stderr: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    hallucination_type: Optional[HallucinationType] = None


@dataclass
class EFLResult:
    """Final result returned by the Execution Feedback Loop.

    Attributes:
        code:       The best code produced (last attempt if all failed).
        passed:     True if at least one attempt passed.
        iterations: Number of generation+execution cycles performed.
        history:    Per-iteration execution results for analysis.
    """

    code: str
    passed: bool
    iterations: int
    history: List[ExecutionResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

def execute_code(
    code: str,
    test_code: str,
    timeout: int = 30,
) -> ExecutionResult:
    """Execute generated code + test code in an isolated subprocess.

    Writes ``code`` followed by ``test_code`` into a temporary ``.py`` file
    and runs it with ``python3``.  The temp file is always removed afterward.

    Args:
        code:      LLM-generated implementation to evaluate.
        test_code: Test harness code that asserts correctness.
        timeout:   Maximum wall-clock seconds before the process is killed.

    Returns:
        ``ExecutionResult`` populated with stdout, stderr, and parsed error
        information.
    """
    combined = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(combined)
        tmp_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            ["python3", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = proc.returncode == 0

        error_type: Optional[str] = None
        error_message: Optional[str] = None
        hallucination_type: Optional[HallucinationType] = None

        if not passed:
            error_type = parse_error_type(proc.stderr)
            error_message = _parse_error_message(proc.stderr)
            if error_type is not None:
                hallucination_type = classify_hallucination(error_type)

        return ExecutionResult(
            passed=passed,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error_type=error_type,
            error_message=error_message,
            hallucination_type=hallucination_type,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False,
            stdout="",
            stderr=f"TimeoutExpired: execution exceeded {timeout}s",
            error_type="TimeoutExpired",
            error_message=f"Execution exceeded {timeout}s",
            hallucination_type=HallucinationType.LOGIC,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Error parsing
# ---------------------------------------------------------------------------

# Matches "ExceptionName:" or bare "ExceptionName" at the start of a line,
# including dotted names like "subprocess.CalledProcessError".
_ERROR_PATTERN = re.compile(
    r"(?m)^([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*)(?::|\s*$)"
)


def parse_error_type(stderr: str) -> Optional[str]:
    """Extract the Python exception class name from a traceback string.

    Scans for the last occurrence of a capitalised identifier followed by a
    colon (standard traceback format).  Scans in reverse to get the outermost
    exception when chained exceptions are present.

    Args:
        stderr: Full stderr / traceback string from the subprocess.

    Returns:
        Exception class name, e.g. ``"ImportError"``, or ``None`` if not found.
    """
    if not stderr.strip():
        return None

    matches = _ERROR_PATTERN.findall(stderr)
    if not matches:
        return None

    for candidate in reversed(matches):
        # Strip module prefix (e.g. "subprocess.CalledProcessError" → "CalledProcessError")
        name = candidate.split(".")[-1]
        if name[0].isupper():
            return name

    return None


def _parse_error_message(stderr: str) -> Optional[str]:
    """Extract the first meaningful error message line from a traceback.

    Args:
        stderr: Full stderr string.

    Returns:
        First non-empty line after the exception class name, or None.
    """
    if not stderr.strip():
        return None

    lines = stderr.strip().splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("File ") and not stripped.startswith("Traceback"):
            return stripped

    return None


def classify_hallucination(error_type: str) -> Optional[HallucinationType]:
    """Map a Python exception class name to a HallucinationType.

    Args:
        error_type: Exception class name, e.g. ``"ImportError"``.

    Returns:
        The corresponding ``HallucinationType``, or ``None`` if unmapped.
    """
    category = ERROR_TO_CATEGORY.get(error_type)
    if category is None:
        return None

    for member in HallucinationType:
        if member.value == category:
            return member

    return None


# ---------------------------------------------------------------------------
# Prompt builder for next-line completion
# ---------------------------------------------------------------------------

def build_completion_prompt(
    cropped_code: str,
    import_statement: str,
    selected_snippets: List[str],
    previous_error: Optional[str] = None,
) -> str:
    """Assemble a prompt for next-line code completion with cross-file context.

    Args:
        cropped_code:      Code written so far in the current file.
        import_statement:  Import statements from the current file.
        selected_snippets: Code snippets from other files, selected by HCCS.
        previous_error:    Error from a previous EFL iteration, if any.

    Returns:
        Formatted prompt string ready for a causal LM.
    """
    parts: List[str] = []

    for snippet in selected_snippets:
        parts.append(f"# Cross-file context:\n{snippet}")

    if previous_error:
        parts.append(f"# Previous attempt failed with:\n# {previous_error}")

    if import_statement.strip():
        parts.append(import_statement)

    parts.append(cropped_code)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Execution Feedback Loop
# ---------------------------------------------------------------------------

def run_efl(
    cropped_code: str,
    import_statement: str,
    contexts: List[Dict[str, str]],
    scores: np.ndarray,
    generate_fn: Callable[[str], str],
    top_k: int = 5,
    max_iterations: int = 3,
    timeout: int = 10,
) -> EFLResult:
    """Run the Execution Feedback Loop for next-line prediction.

    On each iteration:
      1. Select top-k context chunks by current scores.
      2. Build a completion prompt.
      3. Call ``generate_fn`` to get a predicted next line.
      4. Construct a minimal executable test (imports + code + prediction).
      5. Execute in sandbox.
      6. If passed → return immediately.
      7. If failed → classify error, boost scores via type router, retry.

    Args:
        cropped_code:    Code written so far in the current file.
        import_statement: Import statements from the current file.
        contexts:        Full list of context dicts with ``"snippet"`` and
                         ``"path"`` keys.
        scores:          1-D array of HCCS scores for each context chunk.
        generate_fn:     Callable that takes a prompt and returns a predicted
                         next line of code.
        top_k:           Number of top-scoring chunks to include.  Default 5.
        max_iterations:  Maximum generation+execution cycles.  Default 3.
        timeout:         Subprocess timeout per execution attempt.

    Returns:
        ``EFLResult`` with the best prediction, pass/fail, iteration count,
        and per-iteration history.
    """
    current_scores = scores.copy().astype(np.float64)
    history: List[ExecutionResult] = []
    best_code = ""
    previous_error: Optional[str] = None

    for iteration in range(max_iterations):
        # Select top-k chunks
        k = min(top_k, len(contexts))
        top_indices = np.argsort(current_scores)[::-1][:k]
        selected_snippets = [contexts[i]["snippet"] for i in top_indices]

        # Build prompt and generate
        prompt = build_completion_prompt(
            cropped_code, import_statement, selected_snippets, previous_error
        )
        predicted_line = generate_fn(prompt)
        best_code = predicted_line

        # Construct executable test: imports + cropped_code + predicted line
        test_snippet = (
            import_statement + "\n\n"
            + cropped_code + "\n"
            + predicted_line
        )
        result = execute_code(test_snippet, "", timeout=timeout)
        history.append(result)

        if result.passed:
            return EFLResult(
                code=predicted_line,
                passed=True,
                iterations=iteration + 1,
                history=history,
            )

        # Diagnose and boost context scores for retry
        if result.error_type is not None:
            previous_error = (
                f"{result.error_type}: {result.error_message}"
                if result.error_message
                else result.error_type
            )
            boosts = error_boost(result.error_type)
            current_scores = boost_scores(current_scores, contexts, boosts)

    return EFLResult(
        code=best_code,
        passed=False,
        iterations=max_iterations,
        history=history,
    )
