"""
type_router.py — Rule-based mapping from Python exception types to context categories.

Uses the stdlib ``ast`` module to parse repo source files and extract targeted
context (imports, definitions, signatures, tests) without executing any code.

The mapping is intentionally rule-based (not learned) because the relationship
between error type and remediation context is logical and deterministic:
    ImportError  → show what can be imported
    NameError    → show what names exist
    TypeError    → show function signatures
    AssertionError → show what the tests expect

See docs/DECISIONS.md, Decision 3 for the full rationale.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional

from haluguard.hccs import HallucinationType


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

ERROR_TO_CATEGORY: Dict[str, str] = {
    # RESOURCE: missing imports / packages
    "ImportError":          HallucinationType.RESOURCE.value,
    "ModuleNotFoundError":  HallucinationType.RESOURCE.value,

    # NAMING: wrong identifier, attribute, or unbound local
    "NameError":            HallucinationType.NAMING.value,
    "AttributeError":       HallucinationType.NAMING.value,
    "UnboundLocalError":    HallucinationType.NAMING.value,

    # MAPPING: wrong key, index, or argument type
    "KeyError":             HallucinationType.MAPPING.value,
    "IndexError":           HallucinationType.MAPPING.value,
    "TypeError":            HallucinationType.MAPPING.value,

    # LOGIC: wrong values, assertions, or general runtime errors
    "ValueError":           HallucinationType.LOGIC.value,
    "AssertionError":       HallucinationType.LOGIC.value,
    "RuntimeError":         HallucinationType.LOGIC.value,
    "ZeroDivisionError":    HallucinationType.LOGIC.value,
    "RecursionError":       HallucinationType.LOGIC.value,
    "StopIteration":        HallucinationType.LOGIC.value,
}


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------

def _safe_parse(source: str) -> Optional[ast.Module]:
    """Parse a Python source string into an AST.

    Returns ``None`` on ``SyntaxError`` so one malformed file in the repo
    does not crash the entire routing call.

    Args:
        source: Python source code string.

    Returns:
        Parsed ``ast.Module``, or ``None`` if parsing failed.
    """
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _is_python(filepath: str) -> bool:
    """Return True if filepath ends with ``.py``."""
    return filepath.endswith(".py")


# ---------------------------------------------------------------------------
# Context extraction functions
# ---------------------------------------------------------------------------

def fetch_imports(repo_files: Dict[str, str]) -> List[str]:
    """Extract all top-level import statements from every Python file.

    Handles both ``import X`` and ``from X import Y`` forms.  Non-Python
    files and files with syntax errors are skipped silently.

    Useful for remediation of RESOURCE errors (ImportError, ModuleNotFoundError).

    Args:
        repo_files: Mapping of filepath → source code.

    Returns:
        List of import statement strings.  Each entry is prefixed with a
        comment showing the source file, e.g.::

            # weather_app/api.py
            from config import API_KEY, BASE_URL
    """
    results: List[str] = []

    for filepath, source in repo_files.items():
        if not _is_python(filepath):
            continue
        tree = _safe_parse(source)
        if tree is None:
            continue

        file_lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use original source line for fidelity (preserves aliases, etc.)
                line = file_lines[node.lineno - 1].strip()
                results.append(f"# {filepath}\n{line}")

    return results


def fetch_definitions(repo_files: Dict[str, str]) -> List[str]:
    """Extract top-level class and function definition headers.

    Iterates only ``module.body`` (not nested) to focus on the public API
    surface.  For NAMING errors the LLM needs to know what names exist at
    the module level — deeply nested helpers are less relevant.

    Useful for remediation of NAMING errors (NameError, AttributeError).

    Args:
        repo_files: Mapping of filepath → source code.

    Returns:
        List of definition header strings.  Each entry shows the ``def`` /
        ``class`` line plus the first line of its docstring (if present),
        prefixed with a filepath comment.
    """
    results: List[str] = []

    for filepath, source in repo_files.items():
        if not _is_python(filepath):
            continue
        tree = _safe_parse(source)
        if tree is None:
            continue

        file_lines = source.splitlines()

        for node in tree.body:  # module-level only — intentional
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue

            header = file_lines[node.lineno - 1].strip()

            # Grab first docstring line if present
            docstring: Optional[str] = None
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring = node.body[0].value.value.splitlines()[0].strip()

            entry = f"# {filepath}\n{header}"
            if docstring:
                entry += f'\n    """{docstring}"""'
            results.append(entry)

    return results


def fetch_signatures(repo_files: Dict[str, str]) -> List[str]:
    """Extract ALL function signatures (including class methods).

    Unlike ``fetch_definitions``, this uses ``ast.walk`` to recurse into
    classes so method signatures are captured.  For MAPPING errors the LLM
    typically called a real function with the wrong number or type of
    arguments, so all call sites / signatures are relevant.

    Useful for remediation of MAPPING errors (TypeError, KeyError, IndexError).

    Args:
        repo_files: Mapping of filepath → source code.

    Returns:
        List of ``def`` signature lines, prefixed with a filepath comment.
    """
    results: List[str] = []

    for filepath, source in repo_files.items():
        if not _is_python(filepath):
            continue
        tree = _safe_parse(source)
        if tree is None:
            continue

        file_lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig_line = file_lines[node.lineno - 1].strip()
                results.append(f"# {filepath}\n{sig_line}")

    return results


def fetch_tests(repo_files: Dict[str, str]) -> List[str]:
    """Extract complete test function bodies from test files.

    Targets files whose path contains ``"test"`` (covers ``tests/`` folders
    and ``test_*.py`` naming conventions).  Returns the full function body
    so the LLM sees both example inputs and expected outputs.

    Useful for remediation of LOGIC errors (AssertionError, ValueError).

    Args:
        repo_files: Mapping of filepath → source code.

    Returns:
        List of full test-function source strings, prefixed by filepath.
    """
    results: List[str] = []

    for filepath, source in repo_files.items():
        if not _is_python(filepath):
            continue
        if "test" not in filepath.lower():
            continue

        tree = _safe_parse(source)
        if tree is None:
            continue

        file_lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test"):
                continue

            # node.end_lineno is available on Python 3.8+ (safe for 3.9+)
            start = node.lineno - 1
            end = node.end_lineno  # 1-indexed, used as exclusive slice end
            func_text = "\n".join(file_lines[start:end])
            results.append(f"# {filepath}\n{func_text}")

    return results


# ---------------------------------------------------------------------------
# Main routing entry point
# ---------------------------------------------------------------------------

def route(
    error_type: str,
    repo_files: Dict[str, str],
) -> List[str]:
    """Fetch targeted context based on a Python exception type.

    Maps the exception class name to a hallucination category, then
    dispatches to the appropriate extractor.  Falls back to returning all
    context categories when the error type is unrecognised.

    Args:
        error_type: Python exception class name, e.g. ``"ImportError"``.
        repo_files: Mapping of filepath → source code for the target repo.

    Returns:
        List of context strings relevant to resolving the given error.
    """
    category = ERROR_TO_CATEGORY.get(error_type)

    if category == HallucinationType.RESOURCE.value:
        return fetch_imports(repo_files)
    elif category == HallucinationType.NAMING.value:
        return fetch_definitions(repo_files)
    elif category == HallucinationType.MAPPING.value:
        return fetch_signatures(repo_files)
    elif category == HallucinationType.LOGIC.value:
        return fetch_tests(repo_files)
    else:
        # Unknown error type — return everything; HCCS will rank and filter
        context: List[str] = []
        context.extend(fetch_imports(repo_files))
        context.extend(fetch_definitions(repo_files))
        context.extend(fetch_signatures(repo_files))
        context.extend(fetch_tests(repo_files))
        return context
