"""
test_type_router_protocols.py — Protocol compliance of the three routers.

Ensures all routers (``RegexTypeRouter``, ``NoOpTypeRouter``,
``LearnedTypeRouter``) produce boost dicts with every ``HallucinationType``
key populated with a finite float in [0, 1], and that the module-level
back-compat functions still work.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest

from haluguard.hccs import HallucinationType
from haluguard.type_router import (
    AstSymbolRouter,
    LearnedRouterHead,
    LearnedTypeRouter,
    NoOpTypeRouter,
    RegexTypeRouter,
    TypeRouterBase,
    apply_router_boosts,
    boost_scores,
    classify_snippet,
    error_boost,
    normalize_scores,
    predict_boost,
)


HTYPES = [h.value for h in HallucinationType]


def _assert_valid_boost_dict(d: Dict[str, float]) -> None:
    assert set(d.keys()) == set(HTYPES), f"missing keys: {set(HTYPES) - set(d.keys())}"
    for k, v in d.items():
        assert isinstance(v, float), f"{k} is not a float"
        assert math.isfinite(v), f"{k} is non-finite"
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


class TestRegexRouter:
    def test_is_type_router_base(self) -> None:
        r = RegexTypeRouter()
        assert isinstance(r, TypeRouterBase)
        assert r.name == "regex"

    def test_predict_boost_shape(self) -> None:
        r = RegexTypeRouter()
        d = r.predict_boost("obj.method()")
        _assert_valid_boost_dict(d)
        # Method call triggers NAMING boost in the rule table.
        assert d[HallucinationType.NAMING.value] > 0.0

    def test_error_boost_shape(self) -> None:
        r = RegexTypeRouter()
        _assert_valid_boost_dict(r.error_boost("ImportError"))

    def test_module_functions_still_work(self) -> None:
        # Back-compat with tests/test_efl.py.
        _assert_valid_boost_dict(predict_boost("x = y"))
        _assert_valid_boost_dict(error_boost("NameError"))
        assert classify_snippet("import os\nimport sys", "x.py") == HallucinationType.RESOURCE.value


class TestNoOpRouter:
    def test_predict_boost_is_zero(self) -> None:
        r = NoOpTypeRouter()
        d = r.predict_boost("anything at all")
        _assert_valid_boost_dict(d)
        assert all(v == 0.0 for v in d.values())

    def test_error_boost_delegates(self) -> None:
        r = NoOpTypeRouter()
        # EFL path still needs a non-zero signal — delegates to rule table.
        d = r.error_boost("ImportError")
        _assert_valid_boost_dict(d)
        assert d[HallucinationType.RESOURCE.value] > 0.0


class TestAstSymbolRouter:
    def test_predict_boost_shape(self) -> None:
        r = AstSymbolRouter()
        d = r.predict_boost("from service import Client\nresult = client.fetch(user_id)")
        _assert_valid_boost_dict(d)
        assert d[HallucinationType.RESOURCE.value] > 0.0
        assert d[HallucinationType.NAMING.value] > 0.0

    def test_context_boosts_symbol_overlap(self) -> None:
        r = AstSymbolRouter()
        contexts = [
            {"snippet": "class Client:\n    def fetch(self):\n        pass", "path": "client.py"},
            {"snippet": "def unrelated():\n    pass", "path": "other.py"},
        ]
        boosts = r.context_boosts("client = Client()\nclient.fetch()", contexts)
        assert boosts.shape == (2,)
        assert boosts[0] > boosts[1]


class TestLearnedRouterHead:
    def test_forward_shape(self) -> None:
        import torch

        head = LearnedRouterHead(hidden_size=16, num_categories=4)
        out = head(torch.randn(3, 16))
        assert out.shape == (3, 4)


class TestLearnedRouter:
    def test_predict_boost_shape_no_encoder(self) -> None:
        # With no encoder/tokenizer, the router falls back to a zero
        # embedding, which still exercises the head + shape contract.
        head = LearnedRouterHead(hidden_size=32, num_categories=4)
        router = LearnedTypeRouter(
            encoder=None,
            tokenizer=None,
            head=head,
            device="cpu",
        )
        _assert_valid_boost_dict(router.predict_boost("obj.method("))

    def test_error_boost_delegates(self) -> None:
        head = LearnedRouterHead(hidden_size=32, num_categories=4)
        router = LearnedTypeRouter(encoder=None, tokenizer=None, head=head, device="cpu")
        d = router.error_boost("KeyError")
        _assert_valid_boost_dict(d)
        assert d[HallucinationType.MAPPING.value] > 0.0

    def test_save_load_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        head = LearnedRouterHead(hidden_size=32, num_categories=4)
        router = LearnedTypeRouter(encoder=None, tokenizer=None, head=head, device="cpu")
        path = tmp_path / "router.pt"
        router.save(path)
        restored = LearnedTypeRouter.load(path, encoder=None, tokenizer=None, device="cpu")
        _assert_valid_boost_dict(restored.predict_boost("x = foo(1)"))
        assert restored.head.hidden_size == 32


class TestBoostScoresIntegration:
    def test_normalize_scores_keeps_constant_shape(self) -> None:
        scores = np.array([7.0, 7.0], dtype=np.float32)
        out = normalize_scores(scores)
        assert out.shape == scores.shape
        assert np.allclose(out, np.zeros_like(scores))

    def test_apply_router_boosts_noop_preserves_logits(self) -> None:
        scores = np.array([-5.0, 2.0, 7.0], dtype=np.float32)
        contexts = [
            {"snippet": "def a(): pass", "path": "a.py"},
            {"snippet": "def b(): pass", "path": "b.py"},
            {"snippet": "def c(): pass", "path": "c.py"},
        ]
        out = apply_router_boosts(scores, contexts, NoOpTypeRouter(), "x = 1")
        assert np.allclose(out, scores)

    def test_boost_scores_stays_bounded(self) -> None:
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        contexts = [
            {"snippet": "import os\nimport sys", "path": "a.py"},
            {"snippet": "class Foo:\n    pass", "path": "b.py"},
            {"snippet": "assert True", "path": "tests/test.py"},
        ]
        boosts = RegexTypeRouter().error_boost("ImportError")
        out = boost_scores(scores, contexts, boosts)
        assert out.shape == scores.shape
        assert out.max() <= 1.0
        assert out.min() >= 0.0
