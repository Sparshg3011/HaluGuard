# AGENTS.md — HaluGuard Project Context

## What this project is

HaluGuard is an NLP/ML research project implementing a system that reduces code
hallucinations in LLM-generated code.  The core claim: selecting context based on
*hallucination-prevention potential* (rather than similarity to the query) significantly
reduces code hallucinations in repository-level code generation.

**Dataset:** RepoBench v1.1 (`tianyang/repobench_python_v1.1`, split `cross_file_first`).
8,033 examples of cross-file next-line prediction with pre-extracted context chunks
and `gold_snippet_index` labels.

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
| `haluguard/hccs.py` | `HallucinationType` enum + frozen CodeBERT helper + `HCCSScorer` MLP |
| `haluguard/type_router.py` | Pre-emptive context boosting based on code pattern analysis |
| `haluguard/efl.py` | Sandboxed executor + Execution Feedback Loop with score-based re-ranking |
| `haluguard/data_pipeline.py` | Generates contrastive triplets from RepoBench `gold_snippet_index` |
| `haluguard/evaluate.py` | Pure metric computations: exact match, edit similarity, CodeBLEU |
| `haluguard/baselines.py` | Baseline context selection: BM25, cosine, no-context, full, gold-only |
| `haluguard/generate.py` | DeepSeek-Coder wrapper for next-line prediction |
| `haluguard/pipeline.py` | End-to-end pipeline: HCCS scoring + type-router boost + EFL |
| `haluguard/chunker.py` | Splits `{filepath: source}` dict into overlapping text chunks (legacy, not used with RepoBench) |

## Import dependency order (no circular imports)

```
hccs  ←  type_router  ←  efl  ←  pipeline
                                   evaluate  (standalone)
baselines                       ←  pipeline
generate                        ←  pipeline
data_pipeline                      (standalone, uses only hccs enum)
chunker                            (standalone, not imported by main pipeline)
```

## Data

- `data/` is **gitignored** — all generated artefacts live there
- `data/triplets.jsonl` — contrastive training triplets (JSONL, one per line)
- `data/embeddings/` — pre-computed `.pt` CodeBERT embeddings (query, chunk, gold indices)
- `data/results/` — evaluation output JSONL files and metrics table

## Model checkpoints

`.pt` files are gitignored.  Save them to `checkpoints/` (also gitignored) or to
Google Drive when working in Colab.  Reference paths via environment variables or
notebook config cells — never hardcode absolute paths.

## When helping the user

- Explain ML terms in plain language — the user understands Python and basic ML but is not a deep ML engineer
- Prioritise working code over perfect code — timeline is 2 weeks
- Notebooks > scripts — primary workspace is Google Colab
- Always include checkpoint-save cells in notebooks (Colab disconnects without warning)
- Test with 10 examples before scaling up
- Batch size 256 for MLP training
- Temperature 0.07 for InfoNCE, 0.2 for DeepSeek-Coder generation
- Max 5 selected chunks (top-k = 5)
- Max 3 EFL retries
- Truncate CodeBERT input to 512 tokens
- Truncate DeepSeek-Coder input to 2048 tokens
