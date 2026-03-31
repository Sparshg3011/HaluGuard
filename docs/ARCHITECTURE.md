# HaluGuard Architecture — Detailed Component Guide

## System overview

```
OFFLINE (train once):
  RepoBench v1.1 examples → gold_snippet_index labels →
  form (query, gold_chunk, other_chunk) triplets → train MLP scorer

INFERENCE (per query):
  cropped_code + context chunks (from RepoBench)
       │
       ├──→ CodeBERT: embed query + all chunks → 768-dim vectors
       │
       ├──→ HCCS Scorer (MLP): score each chunk → ranked list
       │
       ├──→ Type Router: analyse cropped_code → boost scores pre-emptively
       │
       ├──→ Prompt Builder: merge selected snippets + query → prompt string
       │
       ├──→ Code LLM (DeepSeek-Coder): generate next line
       │
       └──→ EFL: execute → pass? done! → fail? boost scores → retry
```

## Component 1: CodeBERT Embedding

**Purpose:** Convert code strings into fixed-size vectors for comparison.

- **Input:** Code string (cropped_code or context snippet)
- **Output:** 768-dim numpy array (CLS token embedding)
- **Model:** `microsoft/codebert-base` (125M params, FROZEN — never trained)
- **Performance:** ~0.1s CPU, ~0.01s GPU per chunk
- **Truncation:** Max 512 tokens

## Component 2: HCCS Scorer (MLP)

**Purpose:** Score how much a context chunk would prevent hallucinations for a given query.

- **Input:** Concatenated [query_emb, chunk_emb] → 1536-dim vector
- **Output:** Scalar in [0, 1]. Higher = better hallucination-prevention potential
- **Architecture:** Linear(1536, 256) → ReLU → Dropout(0.1) → Linear(256, 1) → Sigmoid
- **Parameters:** ~394K trainable (MLP only; encoder frozen)
- **Training:** InfoNCE loss with tau=0.07 over (gold_chunk, non_gold_chunk) pairs

## Component 3: Type Router (Rule-Based)

**Purpose:** Pre-emptively boost HCCS scores for chunks that match predicted error types.

**Pre-emptive mode (before generation):**
- Analyses `cropped_code` with regex patterns
- Method calls (`obj.method()`) → boost NAMING chunks +0.15
- Import statements → boost RESOURCE chunks +0.1
- Type annotations → boost MAPPING chunks +0.1
- Assertions → boost LOGIC chunks +0.1

**Post-failure mode (EFL retry):**
- Maps actual Python exception to hallucination category
- Applies +0.2 boost to matching chunk category
- ImportError → boost RESOURCE, NameError → boost NAMING, etc.

**Snippet classification:** Lightweight heuristics on content and path:
- Import-heavy snippets or `__init__.py` → RESOURCE
- Class/function definitions → NAMING
- Typed function signatures → MAPPING
- Test files → LOGIC

## Component 4: Prompt Builder

**Purpose:** Assemble a prompt for next-line code completion.

Format:
```
# Cross-file context:
{snippet_1}

# Cross-file context:
{snippet_2}

{import_statement}

{cropped_code}
```

## Component 5: Code Generator

**Purpose:** Generate the next line of code using DeepSeek-Coder.

- **Model:** `deepseek-ai/deepseek-coder-1.3b-base`
- **Temperature:** 0.2 (low for deterministic predictions)
- **Max tokens:** 64 (only need one line)
- **Truncation:** Max 2048 prompt tokens

## Component 6: Execution Feedback Loop (EFL)

**Purpose:** Retry generation with boosted context when prediction fails.

Algorithm:
1. Select top-5 chunks by current scores
2. Build prompt, generate next line
3. Construct executable test (imports + code + prediction)
4. Execute in sandbox subprocess (timeout 10s)
5. If passed → done
6. If failed → classify error → call `error_boost()` → `boost_scores()` → retry
7. Max 3 iterations

## Component 7: Baselines

Selection methods for ablation comparison:

| Method | Description |
|--------|-------------|
| No context | Empty context list (lower bound) |
| BM25 top-5 | Keyword overlap via `rank_bm25` |
| CodeBERT cosine top-5 | Cosine similarity in embedding space |
| Full context | All chunks (may exceed window) |
| Gold only | Oracle — always select gold chunk |

## Component 8: Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Exact Match (EM) | 1.0 if predicted == ground truth (stripped) |
| Edit Similarity (ES) | Character-level SequenceMatcher ratio |
| CodeBLEU | Structural code similarity (via `codebleu` library) |

## Data Flow Summary

```
RepoBench v1.1 (HuggingFace)
    ↓ [Notebook 01: Data Pipeline]
    Load 8,033 cross_file_first examples
    ↓
    Pre-compute CodeBERT embeddings (~2hrs on T4)
    ↓
    Generate triplets using gold_snippet_index (instant)
    ↓
    data/triplets.jsonl (~70K-80K triplets)
    data/embeddings/*.pt (query + chunk embeddings)
    ↓ [Notebook 02: Train HCCS]
    Load pre-computed embeddings
    ↓
    Train MLP scorer:
      - Input: concat[query_emb, chunk_emb] (1536-dim)
      - Loss: InfoNCE (gold_score >> non_gold_score)
      - Optimization: Adam, 10 epochs, batch_size=256
    ↓
    checkpoints/hccs_best.pt (trained MLP weights)
    ↓ [Notebook 03: Evaluation]
    Run 7 ablation conditions:
      no_context, BM25, cosine, full_context, gold_only,
      HCCS_only, HCCS+router
    ↓
    Each: select context → build prompt → DeepSeek-Coder → predicted line
    ↓
    Compare against next_line: EM, ES, CodeBLEU
    ↓
    data/results/ (per-method JSONL + metrics table)
```
