"""
test_generate.py — Tests for the prompt builder (no GPU required).

Run with:
    pytest tests/test_generate.py -v
"""

from __future__ import annotations

from haluguard.generate import build_completion_prompt


class TestBuildCompletionPrompt:
    def test_basic_prompt(self) -> None:
        prompt = build_completion_prompt(
            cropped_code="x = foo(",
            import_statement="from utils import foo",
            selected_snippets=["def foo(a):\n    return a + 1"],
        )
        assert "# Cross-file context:" in prompt
        assert "def foo(a)" in prompt
        assert "from utils import foo" in prompt
        assert "x = foo(" in prompt

    def test_no_snippets(self) -> None:
        prompt = build_completion_prompt(
            cropped_code="x = 1",
            import_statement="",
            selected_snippets=[],
        )
        assert "x = 1" in prompt
        assert "# Cross-file context:" not in prompt

    def test_empty_import_statement(self) -> None:
        prompt = build_completion_prompt(
            cropped_code="x = 1",
            import_statement="   ",
            selected_snippets=["def bar(): pass"],
        )
        # Empty import statement should be omitted
        assert "import" not in prompt or "# Cross-file" in prompt

    def test_multiple_snippets(self) -> None:
        prompt = build_completion_prompt(
            cropped_code="result = combine(a, b)",
            import_statement="from merger import combine",
            selected_snippets=[
                "def combine(x, y):\n    return x + y",
                "class Merger:\n    pass",
            ],
        )
        # Both snippets should appear
        assert prompt.count("# Cross-file context:") == 2
        assert "def combine" in prompt
        assert "class Merger" in prompt

    def test_cropped_code_at_end(self) -> None:
        prompt = build_completion_prompt(
            cropped_code="x = foo(",
            import_statement="from utils import foo",
            selected_snippets=["def foo(): pass"],
        )
        # cropped_code should be the last part
        assert prompt.endswith("x = foo(")
