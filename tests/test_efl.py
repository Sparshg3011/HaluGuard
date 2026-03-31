"""
test_efl.py — Tests for the Execution Feedback Loop and its helpers.

These tests do NOT require a GPU, internet access, or a trained model.
They run purely with stdlib and the haluguard package.

Run with:
    pytest tests/test_efl.py -v
"""

from __future__ import annotations

from typing import List

import numpy as np
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
# run_efl (RepoBench-style: contexts + scores)
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
        contexts = [
            {"snippet": "def helper(): pass", "path": "helper.py"},
        ]
        scores = np.array([0.9])

        result = run_efl(
            cropped_code="# set x\n",
            import_statement="",
            contexts=contexts,
            scores=scores,
            generate_fn=generate_fn,
            max_iterations=3,
        )
        assert isinstance(result, EFLResult)
        assert result.passed is True
        assert result.iterations == 1

    def test_passes_on_second_attempt(self) -> None:
        generate_fn = self._make_generate_fn([
            "x = undefined_var",  # fails: NameError
            "x = 42",            # passes
        ])
        contexts = [
            {"snippet": "x = 42", "path": "constants.py"},
            {"snippet": "import os", "path": "utils.py"},
        ]
        scores = np.array([0.5, 0.3])

        result = run_efl(
            cropped_code="",
            import_statement="",
            contexts=contexts,
            scores=scores,
            generate_fn=generate_fn,
            max_iterations=3,
        )
        assert result.passed is True
        assert result.iterations == 2

    def test_all_attempts_fail(self) -> None:
        generate_fn = self._make_generate_fn(["x = undefined_var_xyz"])
        contexts = [
            {"snippet": "def foo(): pass", "path": "foo.py"},
        ]
        scores = np.array([0.5])

        result = run_efl(
            cropped_code="",
            import_statement="",
            contexts=contexts,
            scores=scores,
            generate_fn=generate_fn,
            max_iterations=3,
        )
        assert result.passed is False
        assert result.iterations == 3
        assert len(result.history) == 3

    def test_history_length_matches_iterations(self) -> None:
        generate_fn = self._make_generate_fn(["x = undefined_var_xyz"])
        contexts = [{"snippet": "pass", "path": "a.py"}]
        scores = np.array([0.5])

        result = run_efl(
            cropped_code="",
            import_statement="",
            contexts=contexts,
            scores=scores,
            generate_fn=generate_fn,
            max_iterations=2,
        )
        assert len(result.history) == result.iterations


# ---------------------------------------------------------------------------
# chunker (quick sanity tests — module still works, just not in main pipeline)
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
# type_router (new API: predict_boost, classify_snippet, boost_scores)
# ---------------------------------------------------------------------------

class TestTypeRouter:
    def test_predict_boost_method_calls(self) -> None:
        from haluguard.type_router import predict_boost

        boosts = predict_boost("result = obj.method()")
        assert boosts["naming"] > 0.0

    def test_predict_boost_imports(self) -> None:
        from haluguard.type_router import predict_boost

        boosts = predict_boost("from config import KEY\n")
        assert boosts["resource"] > 0.0

    def test_predict_boost_no_patterns(self) -> None:
        from haluguard.type_router import predict_boost

        boosts = predict_boost("x = 1")
        assert all(v == 0.0 for v in boosts.values())

    def test_classify_snippet_definitions(self) -> None:
        from haluguard.type_router import classify_snippet

        result = classify_snippet("def fetch_weather(city: str):\n    pass", "api.py")
        assert result == "naming"

    def test_classify_snippet_imports(self) -> None:
        from haluguard.type_router import classify_snippet

        result = classify_snippet("import os\nimport sys\nimport json", "utils.py")
        assert result == "resource"

    def test_classify_snippet_test_file(self) -> None:
        from haluguard.type_router import classify_snippet

        result = classify_snippet("x = 1", "tests/test_api.py")
        assert result == "logic"

    def test_boost_scores_applies_boosts(self) -> None:
        from haluguard.type_router import boost_scores

        scores = np.array([0.5, 0.3])
        contexts = [
            {"snippet": "def foo(): pass", "path": "api.py"},
            {"snippet": "import os\nimport sys", "path": "utils.py"},
        ]
        boosts = {"naming": 0.1, "resource": 0.2, "mapping": 0.0, "logic": 0.0}
        adjusted = boost_scores(scores, contexts, boosts)
        assert adjusted[0] > 0.5  # naming boost applied
        assert adjusted[1] > 0.3  # resource boost applied

    def test_boost_scores_capped_at_one(self) -> None:
        from haluguard.type_router import boost_scores

        scores = np.array([0.95])
        contexts = [{"snippet": "def foo(): pass", "path": "api.py"}]
        boosts = {"naming": 0.5, "resource": 0.0, "mapping": 0.0, "logic": 0.0}
        adjusted = boost_scores(scores, contexts, boosts)
        assert adjusted[0] <= 1.0

    def test_error_boost_known_error(self) -> None:
        from haluguard.type_router import error_boost

        boosts = error_boost("ImportError")
        assert boosts["resource"] == 0.2
        assert boosts["naming"] == 0.0

    def test_error_boost_unknown_error(self) -> None:
        from haluguard.type_router import error_boost

        boosts = error_boost("FictionalError")
        assert all(v > 0.0 for v in boosts.values())


# ---------------------------------------------------------------------------
# evaluate (new metrics: exact_match, edit_similarity)
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_exact_match_identical(self) -> None:
        from haluguard.evaluate import exact_match

        assert exact_match("x = 42", "x = 42") == 1.0

    def test_exact_match_different(self) -> None:
        from haluguard.evaluate import exact_match

        assert exact_match("x = 42", "x = 99") == 0.0

    def test_exact_match_whitespace_stripped(self) -> None:
        from haluguard.evaluate import exact_match

        assert exact_match("  x = 42  ", "x = 42") == 1.0

    def test_edit_similarity_identical(self) -> None:
        from haluguard.evaluate import edit_similarity

        assert edit_similarity("x = 42", "x = 42") == 1.0

    def test_edit_similarity_similar(self) -> None:
        from haluguard.evaluate import edit_similarity

        sim = edit_similarity("x = 42", "x = 43")
        assert 0.5 < sim < 1.0

    def test_edit_similarity_completely_different(self) -> None:
        from haluguard.evaluate import edit_similarity

        sim = edit_similarity("abcdef", "zyxwvu")
        assert sim < 0.5

    def test_compute_metrics_empty(self) -> None:
        from haluguard.evaluate import compute_metrics

        result = compute_metrics([], [])
        assert result["em"] == 0.0
        assert result["es"] == 0.0

    def test_compute_metrics_perfect(self) -> None:
        from haluguard.evaluate import compute_metrics

        result = compute_metrics(["x = 1", "y = 2"], ["x = 1", "y = 2"])
        assert result["em"] == 1.0
        assert result["es"] == 1.0

    def test_compute_metrics_table(self) -> None:
        from haluguard.evaluate import compute_metrics_table

        results = {
            "method_a": [("x = 1", "x = 1"), ("y = 2", "y = 2")],
            "method_b": [("x = 1", "x = 99"), ("y = 2", "y = 99")],
        }
        table = compute_metrics_table(results)
        assert len(table) == 2
        # method_a should have higher EM and be first (sorted desc)
        assert table[0]["method"] == "method_a"
        assert table[0]["em"] == 1.0
        assert table[1]["method"] == "method_b"
        assert table[1]["em"] == 0.0
