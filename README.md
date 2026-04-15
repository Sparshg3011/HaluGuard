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
