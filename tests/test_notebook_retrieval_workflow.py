"""Static validation for the training and pipeline notebooks."""

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
    def test_notebooks_compile(self) -> None:
        """All notebooks must contain syntactically valid Python in their code cells."""
        for name in (
            "01_data_pipeline.ipynb",
            "02_train_hccs.ipynb",
            "03_evaluation.ipynb",
            "04_pipeline_demo.ipynb",
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
    def test_training_notebook_has_all_architectures(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("02_train_hccs.ipynb")))

        # All 6 individual architectures must appear
        for arch in (
            "DualEncoder", "DualEncoderDeep", "ListwiseMLP",
            "PairwiseMLP", "InteractionMLP", "BilinearScorer",
        ):
            assert arch in sources, f"Architecture {arch!r} missing from 02_train_hccs.ipynb"

        # EnsembleScorer trained on top-3
        assert "EnsembleScorer" in sources

    def test_training_notebook_uses_curriculum(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("02_train_hccs.ipynb")))

        assert "curriculum_train_epoch" in sources
        assert "curriculum_warmup" in sources

    def test_training_notebook_saves_checkpoints(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("02_train_hccs.ipynb")))

        assert "torch.save" in sources
        assert "CKPT_DIR" in sources
        assert "best_mrr" in sources

    def test_training_notebook_has_early_stopping(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("02_train_hccs.ipynb")))

        assert "patience" in sources
        assert "patience_counter" in sources


class TestNotebook03:
    def test_evaluation_notebook_has_retrieval_section(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("03_evaluation.ipynb")))

        assert "compute_retrieval_summary" in sources
        assert "retrieval_repobench.json" in sources


class TestNotebook04:
    def test_pipeline_demo_notebook_has_efl(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("04_pipeline_demo.ipynb")))

        assert "run_efl" in sources
        assert "efl_result" in sources

    def test_pipeline_demo_notebook_has_error_analysis(self) -> None:
        sources = "\n".join(_code_sources(_load_notebook("04_pipeline_demo.ipynb")))

        assert "hccs_wins" in sources
        assert "cosine_wins" in sources

