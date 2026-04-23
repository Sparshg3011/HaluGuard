"""
test_evaluate_new_metrics.py — Golden-input tests for the new metrics.

These tests do NOT require the optional ``sacrebleu`` / ``rouge_score``
dependencies to produce meaningful results: metrics whose backing library is
absent return 0.0, and identifier metrics use only the stdlib.
"""

from __future__ import annotations

import math

import pytest

from haluguard.evaluate import (
    ALL_METRICS,
    bleu4,
    chrf_plus,
    compute_metrics,
    first_error_type_hist,
    exact_match_normalized,
    identifier_precision,
    identifier_recall,
    identifier_em,
    identifier_f1,
    identifier_tokens,
    rouge_l,
    syntax_valid,
)


class TestIdentifierTokens:
    def test_basic(self) -> None:
        toks = identifier_tokens("x = foo(y)")
        assert {"x", "foo", "y"} <= toks

    def test_strips_builtins_and_keywords(self) -> None:
        toks = identifier_tokens("for x in range(len(items)): pass")
        assert "for" not in toks  # keyword
        assert "range" not in toks  # builtin
        assert "len" not in toks   # builtin
        assert "x" in toks
        assert "items" in toks

    def test_empty(self) -> None:
        assert identifier_tokens("") == set()
        assert identifier_tokens("   ") == set()


class TestIdentifierEMF1:
    def test_em_exact(self) -> None:
        assert identifier_em("x = foo(y)", "x = foo(y)") == 1.0

    def test_em_rearranged(self) -> None:
        # Same identifier set, different order → EM is set-equality so passes.
        assert identifier_em("foo(x, y)", "y + foo(x)") == 1.0

    def test_f1_partial(self) -> None:
        # pred = {x, foo, y}, ref = {x, foo, z} → tp=2, p=2/3, r=2/3, f1=2/3
        score = identifier_f1("x = foo(y)", "x = foo(z)")
        assert math.isclose(score, 2 / 3, rel_tol=1e-6)

    def test_f1_empty_vs_empty(self) -> None:
        assert identifier_f1("", "") == 1.0

    def test_f1_one_empty(self) -> None:
        assert identifier_f1("", "x") == 0.0

    def test_precision_recall(self) -> None:
        pred = "client.fetch(user_id)"
        ref = "client.fetch(user_id, timeout=timeout)"
        assert identifier_precision(pred, ref) == 1.0
        assert 0.0 < identifier_recall(pred, ref) < 1.0


class TestNormalizedAndSyntaxMetrics:
    def test_normalized_exact_match(self) -> None:
        assert exact_match_normalized("return   x + 1", "return x + 1") == 1.0

    def test_syntax_valid_handles_return_fragment(self) -> None:
        assert syntax_valid("return x + 1") == 1.0
        assert syntax_valid("def broken(:") == 0.0


class TestSurfaceMetrics:
    def test_bleu4_identical(self) -> None:
        score = bleu4("return x + 1", "return x + 1")
        # sacrebleu returns ~1.0 on identical; nltk fallback may be lower on
        # very short strings but still > 0.5.
        assert score > 0.5 or score == 0.0  # 0.0 only when neither lib installed

    def test_chrf_plus_identical(self) -> None:
        score = chrf_plus("return x + 1", "return x + 1")
        # sacrebleu present → ~1.0; absent → 0.0. Accept either.
        assert score > 0.9 or score == 0.0

    def test_rouge_l_identical(self) -> None:
        score = rouge_l("return x + 1", "return x + 1")
        assert score > 0.9 or score == 0.0

    def test_empty_both_is_match(self) -> None:
        assert bleu4("", "") == 1.0
        assert chrf_plus("", "") == 1.0
        assert rouge_l("", "") == 1.0


class TestFirstErrorTypeHist:
    def test_dict_style(self) -> None:
        hist = first_error_type_hist(
            [
                {"history": [{"error_type": "NameError"}]},
                {"history": [{"error_type": "NameError"}]},
                {"history": [{"error_type": None}]},
                {"history": []},
            ]
        )
        assert hist["NameError"] == 2
        assert hist["none"] == 2

    def test_list_of_execresult_like(self) -> None:
        class Fake:
            def __init__(self, error_type):
                self.error_type = error_type

        hist = first_error_type_hist([[Fake("KeyError")], [Fake(None)]])
        assert hist["KeyError"] == 1
        assert hist["none"] == 1


class TestComputeMetricsDispatch:
    def test_default_shape_preserves_legacy(self) -> None:
        out = compute_metrics(["return 1"], ["return 1"])
        assert set(out.keys()) == {"em", "es", "codebleu"}

    def test_explicit_subset(self) -> None:
        out = compute_metrics(
            ["return 1", "return 2"],
            ["return 1", "return 3"],
            metrics=["em", "identifier_f1"],
        )
        assert set(out.keys()) == {"em", "identifier_f1"}
        assert out["em"] == 0.5

    def test_codebert_score_silently_skipped_without_encoder(self) -> None:
        out = compute_metrics(
            ["return 1"],
            ["return 1"],
            metrics=["em", "codebert_score"],
        )
        # codebert_score is present but reported as NA/None when unavailable.
        assert "em" in out
        assert out["codebert_score"] is None

    def test_all_metrics_constant_contains_expected(self) -> None:
        assert "em" in ALL_METRICS
        assert "em_normalized" in ALL_METRICS
        assert "identifier_precision" in ALL_METRICS
        assert "identifier_f1" in ALL_METRICS
        assert "codebert_score" in ALL_METRICS
        assert "chrf_plus" in ALL_METRICS
        assert "syntax_valid" in ALL_METRICS
