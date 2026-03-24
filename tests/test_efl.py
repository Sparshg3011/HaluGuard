"""
test_efl.py — Tests for the Execution Feedback Loop and its helpers.

These tests do NOT require a GPU, internet access, or a trained model.
They run purely with stdlib and the haluguard package.

Run with:
    pytest tests/test_efl.py -v
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from haluguard.efl import (
    EFLResult,
    ExecutionResult,
    classify_hallucination,
    execute_code,
    parse_error_type,
    run_efl,
)
from haluguard.hccs import HallucinationType


# ---------------------------------------------------------------------------
# execute_code
# ---------------------------------------------------------------------------

class TestExecuteCode:
    def test_passing_code(self) -> None:
        result = execute_code(code="x = 1 + 1", test_code="assert x == 2")
        assert result.passed is True
        assert result.error_type is None
        assert result.hallucination_type is None

    def test_import_error(self) -> None:
        result = execute_code(
            code="import nonexistent_package_xyz",
            test_code="",
        )
        assert result.passed is False
        assert result.error_type in ("ImportError", "ModuleNotFoundError")
        assert result.hallucination_type == HallucinationType.RESOURCE

    def test_name_error(self) -> None:
        result = execute_code(
            code="y = undefined_variable_xyz + 1",
            test_code="",
        )
        assert result.passed is False
        assert result.error_type == "NameError"
        assert result.hallucination_type == HallucinationType.NAMING

    def test_type_error(self) -> None:
        result = execute_code(
            code="result = 'hello' + 42",
            test_code="",
        )
        assert result.passed is False
        assert result.error_type == "TypeError"
        assert result.hallucination_type == HallucinationType.MAPPING

    def test_assertion_error(self) -> None:
        result = execute_code(
            code="x = 5",
            test_code="assert x == 99, 'expected 99'",
        )
        assert result.passed is False
        assert result.error_type == "AssertionError"
        assert result.hallucination_type == HallucinationType.LOGIC

    def test_syntax_error_in_code(self) -> None:
        result = execute_code(
            code="def broken(:\n    pass",
            test_code="",
        )
        assert result.passed is False
        assert result.error_type is not None  # SyntaxError

    def test_stdout_captured(self) -> None:
        result = execute_code(
            code="print('hello haluguard')",
            test_code="",
        )
        assert result.passed is True
        assert "hello haluguard" in result.stdout

    def test_temp_file_cleaned_up(self, tmp_path: "pytest.TempPathFactory") -> None:
        import tempfile
        from pathlib import Path

        before = set(Path(tempfile.gettempdir()).glob("*.py"))
        execute_code(code="x = 1", test_code="assert x == 1")
        after = set(Path(tempfile.gettempdir()).glob("*.py"))
        # No new .py files should remain
        assert after.issubset(before | before)  # no additions


# ---------------------------------------------------------------------------
# parse_error_type
# ---------------------------------------------------------------------------

class TestParseErrorType:
    def test_import_error(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "tmp.py", line 1, in <module>\n'
            "    import nonexistent\n"
            "ModuleNotFoundError: No module named 'nonexistent'\n"
        )
        assert parse_error_type(stderr) == "ModuleNotFoundError"

    def test_name_error(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "tmp.py", line 1, in <module>\n'
            "    foo()\n"
            "NameError: name 'foo' is not defined\n"
        )
        assert parse_error_type(stderr) == "NameError"

    def test_empty_stderr(self) -> None:
        assert parse_error_type("") is None
        assert parse_error_type("   ") is None

    def test_chained_exception(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            "  ...\n"
            "ValueError: something wrong\n"
            "\nDuring handling of the above exception, another exception occurred:\n"
            "RuntimeError: wrapped error\n"
        )
        result = parse_error_type(stderr)
        # Should return the outermost (last) exception
        assert result in ("RuntimeError", "ValueError")


# ---------------------------------------------------------------------------
# classify_hallucination
# ---------------------------------------------------------------------------

class TestClassifyHallucination:
    @pytest.mark.parametrize("error_type,expected", [
        ("ImportError",          HallucinationType.RESOURCE),
        ("ModuleNotFoundError",  HallucinationType.RESOURCE),
        ("NameError",            HallucinationType.NAMING),
        ("AttributeError",       HallucinationType.NAMING),
        ("UnboundLocalError",    HallucinationType.NAMING),
        ("KeyError",             HallucinationType.MAPPING),
        ("IndexError",           HallucinationType.MAPPING),
        ("TypeError",            HallucinationType.MAPPING),
        ("ValueError",           HallucinationType.LOGIC),
        ("AssertionError",       HallucinationType.LOGIC),
        ("RuntimeError",         HallucinationType.LOGIC),
    ])
    def test_known_errors(
        self, error_type: str, expected: HallucinationType
    ) -> None:
        assert classify_hallucination(error_type) == expected

    def test_unknown_error_returns_none(self) -> None:
        assert classify_hallucination("FictionalError") is None


# ---------------------------------------------------------------------------
# run_efl
# ---------------------------------------------------------------------------

class TestRunEFL:
    def _make_generate_fn(self, codes: List[str]):
        """Return a generate_fn that yields codes in order."""
        call_count = [0]

        def generate_fn(prompt: str) -> str:
            idx = min(call_count[0], len(codes) - 1)
            call_count[0] += 1
            return codes[idx]

        return generate_fn

    def test_passes_on_first_attempt(self) -> None:
        generate_fn = self._make_generate_fn(["x = 42"])
        result = run_efl(
            query="set x to 42",
            initial_context=[],
            test_code="assert x == 42",
            generate_fn=generate_fn,
            max_iterations=3,
        )
        assert isinstance(result, EFLResult)
        assert result.passed is True
        assert result.iterations == 1

    def test_passes_on_second_attempt(self) -> None:
        # First code fails, second succeeds
        generate_fn = self._make_generate_fn([
            "x = 0",         # fails: assert x == 42
            "x = 42",        # passes
        ])
        result = run_efl(
            query="set x to 42",
            initial_context=[],
            test_code="assert x == 42",
            generate_fn=generate_fn,
            max_iterations=3,
        )
        assert result.passed is True
        assert result.iterations == 2

    def test_all_attempts_fail(self) -> None:
        generate_fn = self._make_generate_fn(["x = 0"])
        result = run_efl(
            query="set x to 42",
            initial_context=[],
            test_code="assert x == 42",
            generate_fn=generate_fn,
            max_iterations=3,
        )
        assert result.passed is False
        assert result.iterations == 3
        assert len(result.history) == 3

    def test_history_length_matches_iterations(self) -> None:
        generate_fn = self._make_generate_fn(["x = 1"])
        result = run_efl(
            query="dummy",
            initial_context=[],
            test_code="assert x == 99",
            generate_fn=generate_fn,
            max_iterations=2,
        )
        assert len(result.history) == result.iterations

    def test_context_grows_on_failure(self) -> None:
        """When repo_files is provided, context should expand after a failure."""
        captured_prompts: List[str] = []

        def generate_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            if len(captured_prompts) == 1:
                return "import nonexistent_xyz"  # will fail with ImportError
            return "x = 1"  # passes

        result = run_efl(
            query="test context growth",
            initial_context=["# initial context"],
            test_code="assert x == 1",
            generate_fn=generate_fn,
            repo_files={"helper.py": "import os\nimport sys"},
            max_iterations=3,
        )
        assert result.passed is True
        # Second prompt should contain more context than the first
        assert len(captured_prompts[1]) > len(captured_prompts[0])


# ---------------------------------------------------------------------------
# chunker (quick sanity tests)
# ---------------------------------------------------------------------------

class TestChunker:
    def test_single_file_small(self) -> None:
        from haluguard.chunker import chunk_repo

        repo = {"api.py": "import requests\ndef fetch(): pass"}
        chunks = chunk_repo(repo)
        assert len(chunks) == 1
        assert "# File: api.py" in chunks[0]
        assert "lines 1-" in chunks[0]

    def test_large_file_splits(self) -> None:
        from haluguard.chunker import chunk_repo

        source = "\n".join(f"line_{i}" for i in range(60))
        repo = {"big.py": source}
        chunks = chunk_repo(repo, max_lines=30, stride=15)
        assert len(chunks) > 1

    def test_chunk_headers_are_1_indexed(self) -> None:
        from haluguard.chunker import chunk_repo

        repo = {"f.py": "a\nb\nc"}
        chunks = chunk_repo(repo)
        assert "lines 1-" in chunks[0]

    def test_stride_greater_than_max_lines_raises(self) -> None:
        from haluguard.chunker import chunk_repo

        with pytest.raises(ValueError, match="stride"):
            chunk_repo({"f.py": "x"}, max_lines=10, stride=20)

    def test_empty_file_skipped(self) -> None:
        from haluguard.chunker import chunk_repo

        repo = {"empty.py": ""}
        chunks = chunk_repo(repo)
        assert chunks == []


# ---------------------------------------------------------------------------
# type_router (quick sanity tests)
# ---------------------------------------------------------------------------

class TestTypeRouter:
    def test_route_import_error(self) -> None:
        from haluguard.type_router import route

        repo = {"api.py": "import requests\nfrom config import KEY"}
        results = route("ImportError", repo)
        assert any("import" in r.lower() for r in results)

    def test_route_name_error(self) -> None:
        from haluguard.type_router import route

        repo = {"api.py": "def fetch_weather(city: str) -> dict:\n    pass"}
        results = route("NameError", repo)
        assert any("fetch_weather" in r for r in results)

    def test_route_unknown_error_returns_all(self) -> None:
        from haluguard.type_router import route

        repo = {
            "api.py": "import requests\ndef fetch(): pass",
            "tests/test_api.py": "def test_fetch():\n    assert True",
        }
        results = route("FictionalError", repo)
        # Should return content from multiple extractors
        assert len(results) > 0

    def test_non_python_files_skipped(self) -> None:
        from haluguard.type_router import fetch_imports

        repo = {
            "api.py": "import requests",
            "README.md": "import fake",  # not Python, should be ignored
        }
        results = fetch_imports(repo)
        assert all("api.py" in r for r in results)


# ---------------------------------------------------------------------------
# evaluate (metric function tests)
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_hallucination_rate_all_pass(self) -> None:
        from haluguard.evaluate import compute_hallucination_rate

        results = [{"hallucinated": False}] * 10
        assert compute_hallucination_rate(results) == 0.0

    def test_hallucination_rate_all_fail(self) -> None:
        from haluguard.evaluate import compute_hallucination_rate

        results = [{"hallucinated": True}] * 4
        assert compute_hallucination_rate(results) == 1.0

    def test_hallucination_rate_mixed(self) -> None:
        from haluguard.evaluate import compute_hallucination_rate

        results = [{"hallucinated": True}] * 2 + [{"hallucinated": False}] * 8
        assert compute_hallucination_rate(results) == pytest.approx(0.2)

    def test_hallucination_rate_empty(self) -> None:
        from haluguard.evaluate import compute_hallucination_rate

        assert compute_hallucination_rate([]) == 0.0

    def test_reduction_ratio(self) -> None:
        from haluguard.evaluate import compute_reduction_ratio

        assert compute_reduction_ratio(0.4, 0.2) == pytest.approx(0.5)
        assert compute_reduction_ratio(0.4, 0.4) == pytest.approx(0.0)
        assert compute_reduction_ratio(0.0, 0.0) == 0.0

    def test_reduction_ratio_invalid_inputs(self) -> None:
        from haluguard.evaluate import compute_reduction_ratio

        with pytest.raises(ValueError):
            compute_reduction_ratio(1.5, 0.2)
        with pytest.raises(ValueError):
            compute_reduction_ratio(0.4, -0.1)

    def test_pass_rate(self) -> None:
        from haluguard.evaluate import compute_pass_rate

        results = [{"passed": True}] * 3 + [{"passed": False}] * 7
        assert compute_pass_rate(results) == pytest.approx(0.3)
