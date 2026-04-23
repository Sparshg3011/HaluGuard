"""
test_cached_eval_matrix.py — Cached evaluation helpers.

These tests use tiny in-memory examples and temporary torch tensors. They do
not download datasets, compute embeddings, or load real model checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import torch

from haluguard.artifacts import (
    cceval_embeddings_available,
    load_cceval_cached_artifacts,
    load_repobench_cached_artifacts,
    repobench_required_paths,
    validate_repobench_cached_artifacts,
)
from haluguard.benchmarks.base import Example
from haluguard.eval_matrix import (
    MethodPlan,
    build_method_plan,
    compute_repobench_retrieval_table,
    fake_generator,
    run_generation_method,
    write_table_files,
)


class _Loader:
    name = "fake"

    def __init__(self) -> None:
        self.examples = [
            Example(
                cropped_code="def f(x):\n    return",
                context_chunks=[
                    {"snippet": "def helper(x):\n    return x", "path": "helpers.py"},
                    {"snippet": "class Client:\n    pass", "path": "client.py"},
                ],
                gold_index=0,
                reference="return",
                metadata={"source_index": 0, "task_id": "t0"},
            )
        ]

    def iter_examples(self, limit: Optional[int] = None) -> Iterator[Example]:
        count = 0
        for ex in self.examples:
            if limit is not None and count >= limit:
                return
            yield ex
            count += 1


def _write_repobench_cache(tmp_path: Path, chunk_count: int = 2) -> None:
    emb = tmp_path / "embeddings"
    emb.mkdir()
    for name, path in repobench_required_paths(tmp_path).items():
        if name in ("gold_indices", "test_indices"):
            torch.save(torch.tensor([0]), path)
        elif name.startswith("chunk_"):
            torch.save([torch.randn(chunk_count, 4)], path)
        else:
            torch.save(torch.randn(1, 4), path)


def _write_cceval_cache(
    tmp_path: Path,
    encoder: str = "unixcoder",
    query_view: str = "full",
) -> None:
    emb = tmp_path / "embeddings"
    emb.mkdir(exist_ok=True)
    torch.save(torch.randn(1, 4), emb / "cceval_query_embeddings.pt")
    torch.save([torch.randn(2, 4)], emb / "cceval_chunk_embeddings.pt")
    meta = {
        "encoder": encoder,
        "backend": encoder,
        "query_view": query_view,
        "n_examples": 1,
        "hidden_size": 4,
    }
    (emb / "cceval_query_embeddings.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (emb / "cceval_chunk_embeddings.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_validate_repobench_cache_reports_missing(tmp_path: Path) -> None:
    try:
        validate_repobench_cached_artifacts(tmp_path)
    except FileNotFoundError as exc:
        assert "Missing required cached RepoBench artifacts" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected FileNotFoundError")


def test_load_repobench_cache_uses_existing_tensors(tmp_path: Path) -> None:
    _write_repobench_cache(tmp_path)
    payload = load_repobench_cached_artifacts(tmp_path)
    assert "query_embeddings" in payload
    assert ("unixcoder", "last3") in payload["query_embeddings"]
    # Guardrail: artifact loading module must not expose embedding functions.
    import haluguard.artifacts as artifacts

    assert "embed_code" not in artifacts.__dict__
    assert "batch_embed" not in artifacts.__dict__


def test_crosscodeeval_missing_embeddings_create_skipped_rows(tmp_path: Path) -> None:
    assert not cceval_embeddings_available(tmp_path)
    plans = build_method_plan("crosscodeeval", tmp_path, checkpoint_dirs=[])
    cosine_rows = [p for p in plans if p.family == "cosine"]
    assert cosine_rows
    assert all(p.status == "skipped" for p in cosine_rows)
    assert all(p.skip_reason == "skipped_missing_cached_embeddings" for p in cosine_rows)


def test_crosscodeeval_only_matching_cosine_variant_becomes_ready(tmp_path: Path) -> None:
    _write_cceval_cache(tmp_path, encoder="unixcoder", query_view="full")
    plans = build_method_plan("crosscodeeval", tmp_path, checkpoint_dirs=[])
    cosine_rows = {p.method: p for p in plans if p.family == "cosine"}
    assert cosine_rows["cosine_unixcoder_full"].status == "ready"
    assert cosine_rows["cosine_unixcoder_last3"].skip_reason == "skipped_incompatible_cached_embeddings"
    assert cosine_rows["cosine_codebert_full"].skip_reason == "skipped_incompatible_cached_embeddings"
    assert cosine_rows["cosine_codebert_last3"].skip_reason == "skipped_incompatible_cached_embeddings"


def test_crosscodeeval_haluguard_rows_require_matching_cached_variant(tmp_path: Path) -> None:
    _write_cceval_cache(tmp_path, encoder="unixcoder", query_view="full")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "dual_encoder_best.pt").write_bytes(b"placeholder")

    plans = build_method_plan("crosscodeeval", tmp_path, checkpoint_dirs=[ckpt_dir])
    row = next(p for p in plans if p.method == "dual_encoder__noop")
    assert row.status == "skipped"
    assert row.skip_reason == "skipped_incompatible_cached_embeddings"


def test_run_generation_method_preserves_skipped_row(tmp_path: Path) -> None:
    plan = MethodPlan(
        benchmark="crosscodeeval",
        method="cosine_unixcoder_last3",
        family="cosine",
        status="skipped",
        skip_reason="skipped_missing_cached_embeddings",
    )
    summary = run_generation_method(
        plan=plan,
        loader=_Loader(),
        generator=fake_generator,
        output_dir=tmp_path,
    )
    row = summary.to_row()
    assert row["status"] == "skipped"
    assert row["skip_reason"] == "skipped_missing_cached_embeddings"
    assert row["n"] == 0


def test_run_generation_method_skips_incompatible_cceval_cache(tmp_path: Path) -> None:
    _write_cceval_cache(tmp_path, encoder="unixcoder", query_view="full")
    artifacts = load_cceval_cached_artifacts(tmp_path)
    assert artifacts is not None

    plan = MethodPlan(
        benchmark="crosscodeeval",
        method="cosine_unixcoder_last3",
        family="cosine",
        status="ready",
        backend="unixcoder",
        query_view="last3",
    )
    summary = run_generation_method(
        plan=plan,
        loader=_Loader(),
        generator=fake_generator,
        output_dir=tmp_path,
        cceval_artifacts=artifacts,
    )
    assert summary.status == "skipped"
    assert summary.skip_reason == "skipped_incompatible_cached_embeddings"


def test_run_generation_method_baseline_writes_details(tmp_path: Path) -> None:
    plan = MethodPlan(benchmark="repobench", method="no_context", family="baseline")
    summary = run_generation_method(
        plan=plan,
        loader=_Loader(),
        generator=fake_generator,
        output_dir=tmp_path,
        repobench_artifacts={"query_embeddings": {}, "chunk_embeddings": {}},
        metrics=["em", "non_empty_rate", "avg_prediction_chars"],
        limit=1,
    )
    row = summary.to_row()
    assert row["status"] == "completed"
    assert row["n"] == 1
    assert row["non_empty_rate"] == 1.0
    assert summary.details_path is not None
    assert Path(summary.details_path).exists()


def test_run_generation_method_filters_oversized_cached_indices(tmp_path: Path) -> None:
    _write_repobench_cache(tmp_path, chunk_count=4)
    artifacts = load_repobench_cached_artifacts(tmp_path)
    plan = MethodPlan(
        benchmark="repobench",
        method="cosine_unixcoder_last3",
        family="cosine",
        backend="unixcoder",
        query_view="last3",
    )
    summary = run_generation_method(
        plan=plan,
        loader=_Loader(),
        generator=fake_generator,
        output_dir=tmp_path,
        repobench_artifacts=artifacts,
        metrics=["em"],
        limit=1,
    )
    assert summary.status == "completed"
    assert summary.details_path is not None
    record = json.loads(Path(summary.details_path).read_text().splitlines()[0])
    assert all(0 <= int(i) < 2 for i in record["selected_indices"])


def test_repobench_retrieval_counts_missing_gold_as_miss(tmp_path: Path) -> None:
    emb = tmp_path / "embeddings"
    emb.mkdir()
    for name, path in repobench_required_paths(tmp_path).items():
        if name in ("gold_indices", "test_indices"):
            torch.save(torch.tensor([4]), path)
        elif name.startswith("chunk_"):
            torch.save([torch.randn(2, 4)], path)
        else:
            torch.save(torch.randn(1, 4), path)

    class _RetrievalLoader:
        name = "repobench"

        def iter_examples(self, limit: Optional[int] = None) -> Iterator[Example]:
            yield Example(
                cropped_code="def f(x):\n    return",
                context_chunks=[
                    {"snippet": "s0", "path": "a.py"},
                    {"snippet": "s1", "path": "b.py"},
                    {"snippet": "s2", "path": "c.py"},
                    {"snippet": "s3", "path": "d.py"},
                    {"snippet": "s4", "path": "e.py"},
                ],
                gold_index=4,
                reference="return",
                metadata={"source_index": 0, "task_id": "r0"},
            )

    artifacts = load_repobench_cached_artifacts(tmp_path)
    plans = [
        MethodPlan(
            benchmark="repobench",
            method="cosine_unixcoder_last3",
            family="cosine",
            status="ready",
            backend="unixcoder",
            query_view="last3",
        )
    ]
    rows = compute_repobench_retrieval_table(
        plans=plans,
        loader=_RetrievalLoader(),
        repobench_artifacts=artifacts,
        checkpoint_dirs=[],
        limit=1,
    )
    assert len(rows) == 1
    assert rows[0]["easy_count"] == 1
    assert rows[0]["easy_acc@1"] == 0.0
    assert rows[0]["easy_acc@3"] == 0.0


def test_write_table_files_preserves_skipped_rows(tmp_path: Path) -> None:
    rows = [
        {"method": "ready", "status": "completed", "em": 1.0},
        {"method": "skip", "status": "skipped", "skip_reason": "missing"},
    ]
    base = tmp_path / "table"
    write_table_files(rows, base)
    loaded = json.loads((tmp_path / "table.json").read_text())
    assert loaded[1]["status"] == "skipped"
    assert (tmp_path / "table.csv").exists()
    assert (tmp_path / "table.md").exists()
