# CLAUDE.md — HaluGuard Project Context

## What this project is

HaluGuard is an NLP/ML research project implementing a system that reduces code
hallucinations in LLM-generated code.  The core claim: selecting context based on
*hallucination-prevention potential* (rather than similarity to the query) significantly
reduces code hallucinations in repository-level code generation.

## Development commands

```bash
# Install (editable mode, includes dev deps)
pip install -e ".[dev]"

# Run tests (no GPU or internet required)
pytest tests/

# Start Jupyter for notebooks
jupyter notebook notebooks/
```

## Coding conventions

- **Python 3.9+** — use `Optional[X]`, `Dict[K, V]`, `List[X]` from `typing`; NOT `X | None` or built-in generic aliases
- **Type hints** on ALL function signatures
- **Docstrings** on all public functions and classes
- **`dataclasses.dataclass`** for data containers (`ExecutionResult`, `ContrastiveTriplet`, etc.)
- **`pathlib.Path`** for all file paths — no bare strings for paths

## Architecture (one-line summary per module)

| Module | Role |
|--------|------|
| `haluguard/chunker.py` | Splits `{filepath: source}` dict into overlapping text chunks |
| `haluguard/hccs.py` | `HallucinationType` enum + frozen CodeBERT helper + `HCCSScorer` MLP |
| `haluguard/type_router.py` | Maps Python error types → context categories via `ast` parsing |
| `haluguard/efl.py` | Sandboxed executor + Execution Feedback Loop |
| `haluguard/data_pipeline.py` | Generates contrastive triplets for HCCS training |
| `haluguard/evaluate.py` | Pure metric computations (no ML imports) |
| `haluguard/pipeline.py` | Thin wrapper wiring all components for inference |

## Import dependency order (no circular imports)

```
hccs  ←  type_router  ←  efl  ←  data_pipeline  ←  pipeline
                                   evaluate       ←  pipeline
chunker                         ←  data_pipeline  ←  pipeline
```

## Data

- `data/` is **gitignored** — all generated artefacts live there
- `data/triplets.jsonl` — contrastive training triplets (JSONL, one per line)
- `data/embeddings/` — pre-computed `.npy` CodeBERT embeddings
- `data/results/` — evaluation output CSVs

## Model checkpoints

`.pt` files are gitignored.  Save them to `checkpoints/` (also gitignored) or to
Google Drive when working in Colab.  Reference paths via environment variables or
notebook config cells — never hardcode absolute paths.

## When helping the user

- Explain ML terms in plain language — the user understands Python and basic ML but is not a deep ML engineer
- Use the weather-app example from `docs/DRY_RUN.md` when illustrating data flow
- Prioritise working code over perfect code — timeline is 2 weeks
- Notebooks > scripts — primary workspace is Google Colab
- Always include checkpoint-save cells in notebooks (Colab disconnects without warning)
- Test with 10 tasks before scaling up
