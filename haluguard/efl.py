"""
efl.py — Execution Feedback Loop (EFL).

Generates code, executes it in a sandboxed subprocess, classifies the error,
fetches targeted context from the repo, and retries — up to max_iterations times.

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

from haluguard.hccs import HallucinationType
from haluguard.type_router import ERROR_TO_CATEGORY, route


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
# Prompt builder (used internally by run_efl)
# ---------------------------------------------------------------------------

def _build_prompt(
    query: str,
    context_chunks: List[str],
    previous_error: Optional[str] = None,
) -> str:
    """Assemble a prompt string from context chunks and the query.

    Args:
        query:          The coding task description.
        context_chunks: List of context strings (from HCCS + router).
        previous_error: Error message from the previous iteration, if any.

    Returns:
        Formatted prompt string ready to send to the code LLM.
    """
    parts: List[str] = ["# Repository context (selected by HaluGuard):\n"]
    parts.extend(context_chunks)

    if previous_error:
        parts.append(f"\n# Previous attempt failed with:\n# {previous_error}\n")

    parts.append(f"\n# Task:\n# {query}\n")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Execution Feedback Loop
# ---------------------------------------------------------------------------

def run_efl(
    query: str,
    initial_context: List[str],
    test_code: str,
    generate_fn: Callable[[str], str],
    repo_files: Optional[Dict[str, str]] = None,
    max_iterations: int = 3,
    timeout: int = 30,
) -> EFLResult:
    """Run the Execution Feedback Loop for one coding task.

    On each iteration:
      1. Build a prompt from the current context.
      2. Call ``generate_fn`` to get a code candidate.
      3. Execute the candidate against ``test_code``.
      4. If passed → return immediately.
      5. If failed → classify the error, fetch targeted context from
         ``repo_files``, append to context, and retry.

    Args:
        query:           Natural-language description of the coding task.
        initial_context: Starting context chunks (from HCCS scorer output).
        test_code:       Test assertions to verify correctness.
        generate_fn:     Callable that takes a prompt string and returns
                         generated Python code as a string.
        repo_files:      Mapping of filepath → source used for targeted
                         context retrieval on failure.  If None, no new
                         context is added on retry.
        max_iterations:  Maximum number of generate+execute cycles.
                         Default 3 (matches the original paper).
        timeout:         Subprocess timeout in seconds per execution attempt.

    Returns:
        ``EFLResult`` with the best code, pass/fail status, iteration count,
        and per-iteration history.
    """
    current_context = list(initial_context)
    history: List[ExecutionResult] = []
    best_code = ""
    previous_error: Optional[str] = None

    for iteration in range(max_iterations):
        prompt = _build_prompt(query, current_context, previous_error)
        code = generate_fn(prompt)
        best_code = code

        result = execute_code(code, test_code, timeout=timeout)
        history.append(result)

        if result.passed:
            return EFLResult(
                code=code,
                passed=True,
                iterations=iteration + 1,
                history=history,
            )

        # Diagnose and fetch targeted remediation context
        if result.error_type is not None:
            previous_error = (
                f"{result.error_type}: {result.error_message}"
                if result.error_message
                else result.error_type
            )
            if repo_files is not None:
                new_context = route(result.error_type, repo_files)
                current_context = current_context + new_context  # append, don't replace

    return EFLResult(
        code=best_code,
        passed=False,
        iterations=max_iterations,
        history=history,
    )
