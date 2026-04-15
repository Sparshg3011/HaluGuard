# HaluGuard

**Execution-Grounded Contrastive Context Selection for Hallucination-Free Repository-Level Code Generation**

An implementation and evaluation of the HaluGuard framework (Gupta et al., USC) for reducing code hallucinations in LLM-generated code.

## Quick start

1. Open `notebooks/01_data_pipeline.ipynb` in Google Colab
2. Follow the setup cells to install dependencies
3. Run through each notebook in order (01 → 02 → 03)

## What this project does

When AI writes code for a large project, it often hallucinates — inventing functions that don't exist, importing nonexistent libraries, or calling APIs with wrong arguments. HaluGuard fixes this by:

1. **Smart context selection** — Instead of giving the AI code that "looks similar" to the task, give it code that actually prevents mistakes (imports, type signatures, variable definitions)
2. **Type-specific routing** — Different errors need different context. Missing imports need `requirements.txt`. Wrong function names need the actual function definitions.
3. **Execution feedback loop** — Run the generated code, read the error, fetch targeted context, try again.

## Project structure

```
├── CLAUDE.md                  # Context for Claude Code
├── README.md
├── pyproject.toml             # Dependencies (pip install -e ".[dev]")
├── docs/
│   ├── ARCHITECTURE.md        # Detailed component descriptions
│   ├── DRY_RUN.md             # Complete worked example
│   └── GLOSSARY.md            # Technical terms explained
├── notebooks/                 # Colab notebooks (run these)
│   ├── utils.py               # Shared Colab utilities
│   ├── 01_data_pipeline.ipynb # Generate contrastive triplets
│   ├── 02_train_hccs.ipynb    # Train the MLP scorer
│   └── 03_evaluation.ipynb    # Run evaluation on benchmarks
├── haluguard/                 # Python package
│   ├── chunker.py             # Repo file → overlapping text chunks
│   ├── hccs.py                # CodeBERT + MLP scorer (HallucinationType enum lives here)
│   ├── type_router.py         # Error type → context category (AST-based)
│   ├── efl.py                 # Execution Feedback Loop + sandbox executor
│   ├── data_pipeline.py       # Contrastive triplet generation
│   ├── pipeline.py            # End-to-end inference pipeline
│   └── evaluate.py            # Hallucination rate metrics
├── tests/
│   └── test_efl.py            # Tests (no GPU required)
└── data/                      # Generated data (gitignored)
    ├── triplets.jsonl
    ├── embeddings/
    └── results/
```

## Installation

```bash
# Editable install with dev dependencies
pip install -e ".[dev]"

# Run tests (no GPU or internet needed)
pytest tests/ -v
```

## Requirements

- Python 3.9+
- Google Colab Pro (free for US students via `.edu` email) or any environment with a T4 GPU
- `torch`, `transformers`, `datasets`, `numpy`, `tqdm`, `rank-bm25`

## Documentation

- **New to this project?** Read `docs/DRY_RUN.md` first — it walks through one complete example.
- **Want to understand the architecture?** Read `docs/ARCHITECTURE.md`.
- **Confused by a term?** Check `docs/GLOSSARY.md`.
- **Using Claude Code?** The `CLAUDE.md` file has all the context.

## Implementation status

| Component | Status | Location |
|-----------|--------|----------|
| Chunker | Complete | `haluguard/chunker.py` |
| Type router | Complete | `haluguard/type_router.py` |
| Sandbox executor | Complete | `haluguard/efl.py` |
| Execution Feedback Loop | Complete | `haluguard/efl.py` |
| Metrics | Complete | `haluguard/evaluate.py` |
| HCCS scorer architecture | Complete | `haluguard/hccs.py` |
| HCCS training loop | Notebook | `notebooks/02_train_hccs.ipynb` |
| Data pipeline | Notebook | `notebooks/01_data_pipeline.ipynb` |
| Full evaluation | Notebook | `notebooks/03_evaluation.ipynb` |

## Based on

Gupta, S., Liang, A., Hancock, K., Ho, D., & Liang, F. (2025). HaluGuard: Execution-Grounded Contrastive Context Selection for Hallucination-Free Repository-Level Code Generation. University of Southern California.
# HaluGuard

**Execution-Grounded Contrastive Context Selection for Hallucination-Free Repository-Level Code Generation**

`Python 3.9+` `PyTorch` `Transformers` `RepoBench v1.1` `CrossCodeEval` `Google Colab`

[Features](#features) · [Architecture](#architecture) · [Quick-Start](#quick-start) · [Benchmarks](#benchmarks) · [Project-Structure](#project-structure) · [Documentation](#documentation)

## About

HaluGuard is a notebook-first research codebase implementing the HaluGuard framework from Gupta et al. for reducing code hallucinations in repository-level code generation.

Instead of retrieving code that merely looks similar to the prompt, HaluGuard ranks context by **hallucination-prevention potential**: imports, signatures, definitions, and other snippets that help an LLM avoid inventing APIs, modules, or method calls. The system combines learned context ranking, type-aware routing, and an execution feedback loop that retries generation after real runtime failures.

This repository includes the training pipeline, evaluation workflow, scorer model zoo, retrieval baselines, benchmark helpers, and Colab-ready notebooks used to reproduce the full workflow end to end.

## Features

| Feature | Description |
|---------|-------------|
| Hallucination-aware context selection | Learns to rank repository chunks by how well they prevent generation mistakes, rather than by surface similarity alone |
| Seven scorer architectures | Includes `DualEncoder`, `DualEncoderDeep`, `ListwiseMLP`, `PairwiseMLP`, `InteractionMLP`, `BilinearScorer`, and `EnsembleScorer` |
| Type-specific routing | Detects likely failure modes such as missing imports, wrong names, bad argument mappings, and logic errors, then boosts matching context |
| Execution feedback loop | Runs generated code in a sandbox, reads the error signal, re-ranks context, and retries generation up to 3 times |
| Strong baselines | Compare against BM25, cosine retrieval, lexical baselines, random ranking, no-context, full-context, and gold-only oracle selection |
| Retrieval benchmark support | Evaluates both `full` and `last3` query views with easy/hard candidate buckets, accuracy@k, and MRR |
| Colab-first workflow | Four notebooks cover data prep, training, evaluation, and an end-to-end pipeline demo |
| Tested core components | Includes CPU-friendly tests for baselines, training helpers, metrics, notebook integrity, retrieval benchmarking, and EFL behavior |

## Architecture

### How It Works

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. EMBED    │────▶│   2. RANK    │────▶│  3. ROUTE    │────▶│ 4. GENERATE  │────▶│  5. VERIFY   │
│              │     │              │     │              │     │              │     │              │
│ Encode query │     │ Score repo   │     │ Boost chunks │     │ Ask code LLM │     │ Execute code │
│ and context  │     │ chunks with  │     │ based on     │     │ with top-k   │     │ and retry on │
│ with frozen  │     │ HCCS scorer  │     │ error risk   │     │ context      │     │ failure      │
│ encoders     │     │              │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Pipeline Summary

1. `notebooks/01_data_pipeline.ipynb` loads benchmark examples, computes embeddings, and builds training data from `gold_snippet_index`.
2. `haluguard/models.py` provides the scorer architectures used to rank candidate context chunks.
3. `haluguard/type_router.py` boosts context likely to prevent specific hallucination types.
4. `haluguard/generate.py` wraps next-line code generation with DeepSeek-Coder.
5. `haluguard/efl.py` executes the generated code, classifies failures, and triggers targeted retries.
6. `haluguard/evaluate.py` and `haluguard/retrieval_benchmark.py` measure generation quality and retrieval quality separately.

### Scorer Architectures

| Model | Approx. Params | Key Idea | Training Objective |
|-------|----------------|----------|--------------------|
| `DualEncoder` | ~200K | Cosine similarity with learned temperature | Listwise cross-entropy |
| `DualEncoderDeep` | ~800K | Two-layer projection before similarity scoring | Listwise cross-entropy |
| `ListwiseMLP` | ~400K | Concatenate query and chunk embeddings, then rank directly | Listwise cross-entropy |
| `PairwiseMLP` | ~400K | Pairwise scoring over concatenated embeddings | InfoNCE |
| `InteractionMLP` | ~1.1M | ESIM-style interaction features with higher-capacity MLP | Listwise cross-entropy |
| `BilinearScorer` | ~98K | Factorized bilinear scoring with low capacity | Listwise cross-entropy |
| `EnsembleScorer` | Tiny gate | Learned mixture over the top-performing models | Listwise cross-entropy |

Legacy `HCCSScorer` checkpoints remain supported for backward compatibility.

## Quick Start

### Recommended Workflow: Google Colab

1. Open `notebooks/01_data_pipeline.ipynb` in Google Colab.
2. Run the setup cells to install dependencies and mount storage if needed.
3. Execute the notebooks in order:
   - `01_data_pipeline.ipynb`
   - `02_train_hccs.ipynb`
   - `03_evaluation.ipynb`
   - `04_pipeline_demo.ipynb`
4. Save checkpoints to `checkpoints/` or Google Drive so training progress survives Colab disconnects.

### Local Development

#### Prerequisites

- Python `3.9+`
- A GPU-backed environment for embedding generation and model training
- Optional: Google Colab Pro or another environment with a T4-class GPU

#### Install

```bash
pip install -e ".[dev]"
```

If you want full CodeBLEU support during evaluation:

```bash
pip install -e ".[dev,codebleu]"
```

#### Run Tests

```bash
pytest tests/ -v
```

#### Launch Notebooks Locally

```bash
jupyter notebook notebooks/
```

### Minimal Python Usage

After training a checkpoint in notebook `02_train_hccs.ipynb`, you can load the pipeline directly:

```python
from pathlib import Path

from haluguard.pipeline import HaluGuardPipeline


def generate_fn(prompt: str) -> str:
    # Replace with your own model call.
    raise NotImplementedError


pipeline = HaluGuardPipeline.from_checkpoint(
    Path("checkpoints/hccs_best.pt"),
    top_k=5,
    verbose=True,
)

result = pipeline.run(
    cropped_code=example["cropped_code"],
    import_statement=example["import_statement"],
    contexts=example["context"],
    generate_fn=generate_fn,
    max_iterations=3,
)

print(result["prediction"])
```

## Benchmarks

| Benchmark | Role in HaluGuard |
|-----------|-------------------|
| `tianyang/repobench_python_v1.1` (`cross_file_first`) | Primary training and evaluation benchmark for repository-level next-line prediction with `gold_snippet_index` labels |
| `ArtifactAI/cceval` (Python split) | Zero-shot transfer benchmark for cross-file code completion |

### Evaluation Protocol

- **Generation metrics:** Exact Match (EM), Edit Similarity (ES), and CodeBLEU
- **Retrieval metrics:** Accuracy@k and Mean Reciprocal Rank (MRR)
- **Query views:** `full` cropped code and `last3` in-file lines
- **Candidate buckets:** `easy` (5-9 chunks) and `hard` (10+ chunks)

## Project Structure

```text
├── AGENTS.md                       # Project instructions and coding conventions
├── CLAUDE.md                       # Context file for Claude Code
├── README.md
├── pyproject.toml                  # Package metadata and dependencies
├── docs/
│   ├── ARCHITECTURE.md             # Component-level architecture walkthrough
│   ├── DRY_RUN.md                  # One complete worked example through the system
│   └── GLOSSARY.md                 # Plain-English definitions of technical terms
├── notebooks/
│   ├── 01_data_pipeline.ipynb      # Build embeddings and training examples
│   ├── 02_train_hccs.ipynb         # Train the scorer model zoo
│   ├── 03_evaluation.ipynb         # Run generation and retrieval evaluation
│   ├── 04_pipeline_demo.ipynb      # End-to-end demo and error analysis
│   └── utils.py                    # Shared Colab helpers
├── haluguard/
│   ├── hccs.py                     # Hallucination enum, embeddings, legacy scorer
│   ├── models.py                   # All scorer architectures and registry
│   ├── training.py                 # Training helpers, curriculum, losses, dataloaders
│   ├── type_router.py              # Heuristic routing and score boosts
│   ├── efl.py                      # Sandboxed execution and retry loop
│   ├── baselines.py                # Retrieval and ablation baselines
│   ├── generate.py                 # DeepSeek-Coder generation helpers
│   ├── pipeline.py                 # End-to-end inference pipeline
│   ├── evaluate.py                 # Generation metrics and benchmark utilities
│   └── retrieval_benchmark.py      # Retrieval ranking and summary helpers
├── tests/
│   ├── test_baselines.py
│   ├── test_data_pipeline.py
│   ├── test_efl.py
│   ├── test_evaluate.py
│   ├── test_generate.py
│   ├── test_hccs_training.py
│   └── test_retrieval_benchmark.py
└── data/                           # Generated artifacts (gitignored)
    ├── triplets.jsonl
    ├── embeddings/
    └── results/
```

## Documentation

- **Start here:** [`docs/DRY_RUN.md`](docs/DRY_RUN.md) walks through one example from retrieval to final prediction.
- **Understand the system:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explains the pipeline components in detail.
- **Need plain-English definitions:** [`docs/GLOSSARY.md`](docs/GLOSSARY.md) explains the terminology.
- **Want the full workflow:** run the notebooks in `notebooks/` from `01` through `04`.

## Research Notes

- The project centers on **hallucination prevention**, not just semantic similarity.
- The recommended top-k context size is **5 chunks**.
- The execution feedback loop retries up to **3 times**.
- CodeBERT inputs are truncated to **512 tokens**.
- Generation prompts are truncated to **2048 tokens** for DeepSeek-Coder usage.

## Citation

If you use or adapt this repository, cite the project as:

```text
Gupta, S., Liang, A., Hancock, K., Ho, D., & Liang, F. (2025).
HaluGuard: Execution-Grounded Contrastive Context Selection for
Hallucination-Free Repository-Level Code Generation.
University of Southern California.
```

[Back to Top](#haluguard)
