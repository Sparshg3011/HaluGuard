"""Static validation for the retrieval-benchmark notebook workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def _load_notebook(name: str) -> Dict[str, object]:
    path = NOTEBOOK_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def _code_sources(notebook: Dict[str, object]) -> List[str]:
    cells = notebook["cells"]
    return [
        "".join(cell["source"])
        for cell in cells
        if cell.get("cell_type") == "code"
    ]


class TestNotebookCompilation:
    def test_retrieval_notebooks_compile(self) -> None:
        for name in (
            "01_data_pipeline.ipynb",
            "02_train_hccs.ipynb",
            "04_retrieval_benchmark.ipynb",
        ):
            notebook = _load_notebook(name)
            for source in _code_sources(notebook):
                lines = [
                    line for line in source.splitlines()
                    if not line.lstrip().startswith("!")
                ]
                cleaned = "\n".join(lines).strip()
                if not cleaned:
                    continue
                compile(cleaned, name, "exec")


class TestNotebook01:
    def test_data_pipeline_notebook_requests_expected_variants(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("01_data_pipeline.ipynb")))

        assert "('codebert', QUERY_VIEW_FULL)" in sources
        assert "('codebert', QUERY_VIEW_LAST3)" in sources
        assert "('unixcoder', QUERY_VIEW_LAST3)" in sources
        assert "('sfr400m', QUERY_VIEW_FULL)" in sources

    def test_data_pipeline_notebook_uses_backend_metadata_and_legacy_aliases(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("01_data_pipeline.ipynb")))

        assert "'artifact_type': 'query_embeddings'" in sources
        assert "'artifact_type': 'chunk_embeddings'" in sources
        assert "_save_legacy_codebert_aliases" in sources
        assert "'pooling': 'cls'" in sources


class TestNotebook02:
    def test_training_notebook_selects_checkpoints_by_configurable_metric(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("02_train_hccs.ipynb")))

        assert "CHECKPOINT_METRIC" in sources
        assert "best_val_metric" in sources
        assert "scheduler.step(metrics[CHECKPOINT_METRIC])" in sources
        assert "if metrics[CHECKPOINT_METRIC] > best_val_metric" in sources
        assert "save_checkpoint(scorer, checkpoint_name" in sources

    def test_training_notebook_reloads_best_checkpoint_before_analysis(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("02_train_hccs.ipynb")))

        assert "best_state_dict = load_checkpoint(run['checkpoint_name']" in sources
        assert "scorer.load_state_dict(best_state_dict)" in sources
        assert "Sample validation rankings (best checkpoint loaded)" in sources

    def test_training_notebook_reports_ranking_safe_metrics(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("02_train_hccs.ipynb")))

        assert "metrics['acc@1']" in sources
        assert "metrics['acc@3']" in sources
        assert "metrics['acc@5']" in sources
        assert "metrics['mrr']" in sources


class TestNotebook04:
    def test_benchmark_notebook_filters_to_easy_and_hard_examples(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("04_retrieval_benchmark.ipynb")))

        assert "bucket = get_candidate_bucket(candidate_count)" in sources
        assert "if bucket is None or gold_index < 0 or gold_index >= candidate_count:" in sources

    def test_benchmark_notebook_contains_strict_and_enhanced_rows(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("04_retrieval_benchmark.ipynb")))

        assert "STRICT_METHOD_ORDER" in sources
        assert "'Random'" in sources
        assert "'Jaccard'" in sources
        assert "'Edit'" in sources
        assert "'CodeBERT cosine last3'" in sources
        assert "'UniXcoder cosine last3'" in sources
        assert "'HCCS-CodeBERT-last3'" in sources
        assert "ENHANCED_METHOD_ORDER" in sources
        assert "'CodeBERT cosine full'" in sources
        assert "'UniXcoder cosine full'" in sources
        assert "'HCCS-CodeBERT-full'" in sources
        assert "'HCCS-UniXcoder-full'" in sources

    def test_benchmark_notebook_saves_rankings_and_tables(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("04_retrieval_benchmark.ipynb")))

        assert "save_rankings_jsonl(strict_rankings" in sources
        assert "save_table_json(strict_table" in sources
        assert "save_rankings_jsonl(enhanced_rankings" in sources
        assert "save_table_json(enhanced_table" in sources
