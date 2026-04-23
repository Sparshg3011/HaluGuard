"""
test_run_eval.py — Crash-safe runner behaviour.

Uses a fake loader + fake pipeline to exercise the runner without any models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from haluguard.benchmarks.base import Example
from haluguard.run_eval import run_benchmark


class _FakeLoader:
    """Minimal loader that yields a fixed list of Examples."""

    name = "fakebench"

    def __init__(self, examples: List[Example]) -> None:
        self._examples = examples

    def iter_examples(self, limit: Optional[int] = None) -> Iterator[Example]:
        count = 0
        for ex in self._examples:
            if limit is not None and count >= limit:
                return
            yield ex
            count += 1


class _FakePipeline:
    """A stand-in pipeline that skips all ML and returns canned selections."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.encoder = None
        self.device = "cpu"
        self.top_k = 2
        self.scorer_name = "FakeScorer"

        class _FakeRouter:
            name = "regex"

            def predict_boost(self, _code: str) -> Dict[str, float]:
                return {}

            def error_boost(self, _err: str) -> Dict[str, float]:
                return {}

        self.router = _FakeRouter()


def _make_examples(n: int = 3) -> List[Example]:
    return [
        Example(
            cropped_code=f"def f{i}(x):\n    return",
            context_chunks=[{"snippet": "helper", "path": "u.py"}],
            gold_index=0,
            reference=f"    return {i}",
            language="python",
            import_statement="",
            metadata={"task_id": f"t-{i}", "source_index": i},
        )
        for i in range(n)
    ]


def test_run_benchmark_happy_path(tmp_path: Path) -> None:
    # Monkey-patch the two helpers that would otherwise hit torch/HF.
    import haluguard.run_eval as run_eval_mod

    def fake_select_and_prompt(pipeline, example, **kwargs):
        return f"PROMPT::{example.metadata['source_index']}", [0]

    def fake_generator(prompts: List[str]) -> List[str]:
        return [p.replace("PROMPT::", "    return ") for p in prompts]

    orig = run_eval_mod._select_and_prompt
    run_eval_mod._select_and_prompt = fake_select_and_prompt
    try:
        loader = _FakeLoader(_make_examples(3))
        summary = run_benchmark(
            loader=loader,
            pipeline=_FakePipeline(),  # type: ignore[arg-type]
            generator=fake_generator,
            metrics=["em", "es"],
            limit=3,
            batch_size=2,
            use_efl=False,
            cell_id="happy",
            resume_dir=tmp_path,
            progress_every=100,
        )
    finally:
        run_eval_mod._select_and_prompt = orig

    assert summary["n"] == 3
    assert summary["n_errors"] == 0
    assert "em" in summary["metrics"]
    assert (tmp_path / "happy.jsonl").exists()
    assert (tmp_path / "happy.json").exists()


def test_run_benchmark_resumes(tmp_path: Path) -> None:
    import haluguard.run_eval as run_eval_mod

    def fake_select_and_prompt(pipeline, example, **kwargs):
        return f"PROMPT::{example.metadata['source_index']}", [0]

    call_log: List[int] = []

    def fake_generator(prompts: List[str]) -> List[str]:
        call_log.extend(len(p) for p in prompts)
        return [p.replace("PROMPT::", "    return ") for p in prompts]

    # Pre-populate one record so the runner should skip index 0.
    jsonl = tmp_path / "resume.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "source_index": 0,
                "task_id": "t-0",
                "prediction": "    return 0",
                "reference": "    return 0",
                "selected_indices": [0],
                "gold_index": 0,
                "history": [],
            }
        )
        + "\n"
    )

    orig = run_eval_mod._select_and_prompt
    run_eval_mod._select_and_prompt = fake_select_and_prompt
    try:
        summary = run_benchmark(
            loader=_FakeLoader(_make_examples(3)),
            pipeline=_FakePipeline(),  # type: ignore[arg-type]
            generator=fake_generator,
            metrics=["em"],
            limit=3,
            batch_size=2,
            use_efl=False,
            cell_id="resume",
            resume_dir=tmp_path,
            progress_every=100,
        )
    finally:
        run_eval_mod._select_and_prompt = orig

    # Only 2 new examples should have been generated (index 0 was skipped).
    assert summary["n"] == 3
    # The generator was called at most twice-but-one-batch: 2 new prompts.
    assert sum(1 for _ in call_log) == 2


def test_run_benchmark_swallows_per_example_errors(tmp_path: Path) -> None:
    import haluguard.run_eval as run_eval_mod

    def fake_select_and_prompt(pipeline, example, **kwargs):
        if example.metadata["source_index"] == 1:
            raise RuntimeError("boom")
        return f"PROMPT::{example.metadata['source_index']}", [0]

    def fake_generator(prompts: List[str]) -> List[str]:
        return ["    return 0"] * len(prompts)

    orig = run_eval_mod._select_and_prompt
    run_eval_mod._select_and_prompt = fake_select_and_prompt
    try:
        summary = run_benchmark(
            loader=_FakeLoader(_make_examples(3)),
            pipeline=_FakePipeline(),  # type: ignore[arg-type]
            generator=fake_generator,
            metrics=["em"],
            limit=3,
            batch_size=4,
            use_efl=False,
            cell_id="errors",
            resume_dir=tmp_path,
            progress_every=100,
        )
    finally:
        run_eval_mod._select_and_prompt = orig

    # One example failed, two succeeded — the run didn't crash.
    assert summary["n_errors"] == 1
    assert summary["n"] == 2
