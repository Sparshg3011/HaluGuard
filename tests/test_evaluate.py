"""
test_evaluate.py — Tests for evaluation helpers (generation + retrieval metrics).

Run with:
    pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict
import warnings

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATE_PATH = ROOT / "haluguard" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("haluguard_evaluate_test", EVALUATE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE_MODULE)


def test_compute_metrics_empty_includes_codebleu() -> None:
    """Empty metric batches should still expose a CodeBLEU field."""
    result = EVALUATE_MODULE.compute_metrics([], [])
    assert result["em"] == 0.0
    assert result["es"] == 0.0
    assert result["codebleu"] == 0.0


def test_compute_codebleu_missing_dependency_warns() -> None:
    """Missing optional CodeBLEU dependency should emit a helpful warning."""
    if importlib.util.find_spec("codebleu") is not None:
        pytest.skip("codebleu is installed in this environment")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = EVALUATE_MODULE.compute_codebleu(["x = 1"], ["x = 1"])

    assert score == 0.0
    assert any("CodeBLEU is unavailable" in str(w.message) for w in caught)
