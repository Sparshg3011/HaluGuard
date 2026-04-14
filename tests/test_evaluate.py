"""
test_evaluate.py — Tests for evaluation helpers (generation + retrieval metrics).

Run with:
    pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATE_PATH = ROOT / "haluguard" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("haluguard_evaluate_test", EVALUATE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE_MODULE)
