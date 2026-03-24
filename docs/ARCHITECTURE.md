# HaluGuard Architecture — Detailed Component Guide

## System overview

```
OFFLINE (train once):
  CodeHaluEval tasks → try different context subsets → generate code → execute →
  form (query, good_context, bad_context) triplets → train MLP scorer

INFERENCE (per query):
  Query + Repo files
       │
       ├──→ Chunker: split repo into code chunks
       │
       ├──→ CodeBERT: embed query + all chunks → 768-dim vectors
       │
       ├──→ HCCS Scorer (MLP): score each chunk → ranked list
       │
       ├──→ Type Router: predict error type → add targeted context
       │
       ├──→ Prompt Builder: merge selected chunks + query → prompt string
       │
       ├──→ Code LLM: generate Python code
       │
       └──→ EFL: execute → pass? done! → fail? classify error → fetch more → retry
```

## Component 1: Chunker

**Purpose:** Split repository files into manageable pieces the LLM can consume.

**Input:** `dict[str, str]` — mapping of filepath → file content
```python
{
    "weather_app/api.py": "import requests\nfrom config import API_KEY...",
    "weather_app/config.py": "API_KEY = 'abc123'\nBASE_URL = ...",
    "weather_app/models.py": "from dataclasses import dataclass\n...",
    "weather_app/tests/test_models.py": "from models import WeatherReport\n..."
}
```

**Output:** `list[str]` — list of code chunks, each with a header comment showing origin
```python
[
    "# File: weather_app/api.py (lines 1-10)\nimport requests\nfrom config import API_KEY...",
    "# File: weather_app/config.py (lines 1-5)\nAPI_KEY = 'abc123'\nBASE_URL = ...",
    "# File: weather_app/models.py (lines 1-15)\nfrom dataclasses import dataclass\n...",
    "# File: weather_app/tests/test_models.py (lines 1-8)\nfrom models import WeatherReport\n..."
]
```

**Logic:** Split each file into overlapping windows of ~30 lines with 15-line stride. Small files become single chunks. Each chunk gets a header comment with filepath and line numbers.

**Implementation notes:**
- For repos with <50 files, just use one chunk per file
- For larger repos, use 30-line windows with 50% overlap
- Always include the filepath in the chunk header — the LLM uses this to generate correct imports

---

## Component 2: CodeBERT Embedding

**Purpose:** Convert code text into fixed-size numerical vectors (embeddings) that capture semantic meaning.

**Input:** A string of code (a query or a chunk)
```python
"Write get_forecast that fetches weather and returns WeatherReport"
```

**Output:** A numpy array of shape `(768,)` — 768 floating-point numbers
```python
array([0.0234, -0.1456, 0.3891, ..., 0.0567])  # 768 numbers
```

**How it works:**
1. Tokenize the code string into subword tokens using CodeBERT's tokenizer
2. Feed tokens through the CodeBERT model (125M parameter transformer)
3. Take the CLS token output (first token) as the embedding

**Implementation:**
```python
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()

def embed(text: str) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[0, 0, :].numpy()  # CLS token, shape (768,)
```

**Performance:**
- ~0.1s per chunk on CPU
- ~0.01s per chunk on T4 GPU
- Pre-compute once, save as .npy: `np.save("embeddings.npy", all_embeddings)`

---

## Component 3: HCCS Scorer (MLP)

**Purpose:** Given a (query, chunk) pair, output a score from 0 to 1 indicating how well this chunk prevents hallucinations.

**Input:** Two embeddings concatenated
```python
query_emb = np.array([...])   # shape (768,)
chunk_emb = np.array([...])   # shape (768,)
combined = np.concatenate([query_emb, chunk_emb])  # shape (1536,)
```

**Output:** A single float — the prevention score
```python
0.94  # high = this chunk prevents hallucinations for this query
0.23  # low = this chunk is not helpful
```

**Architecture:**
```
Input (1536) → Linear(1536, 256) → ReLU → Dropout(0.1) → Linear(256, 1) → Sigmoid
```
Total trainable parameters: ~394K (1536*256 + 256 + 256*1 + 1)

**Training data:** Contrastive triplets `(query_emb, pos_chunk_emb, neg_chunk_emb)`
- `pos_chunk_emb` = chunk that led to passing code
- `neg_chunk_emb` = chunk that led to hallucination

**Training loss (InfoNCE):**
```python
pos_score = model(concat(query_emb, pos_chunk_emb))  # should be high
neg_score = model(concat(query_emb, neg_chunk_emb))  # should be low
loss = -log(exp(pos_score/tau) / (exp(pos_score/tau) + exp(neg_score/tau)))
```
Where `tau = 0.07` is a temperature parameter.

**At inference:** Score every chunk, sort descending, take top-k (typically k=5).
```python
scores = [model(concat(query_emb, chunk_emb)) for chunk_emb in all_chunk_embs]
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
selected_chunks = [chunks[i] for i in top_indices]
```

---

## Component 4: Type Router

**Purpose:** Add extra context targeted at the most likely hallucination type.

**Two modes:**

### Pre-generation mode (prediction)
Uses the type classifier (trained alongside HCCS) to predict which error is most likely:
```python
query_emb → Linear(768, 128) → ReLU → Linear(128, 4) → Softmax
→ [mapping=0.30, naming=0.45, resource=0.18, logic=0.07]
→ Highest: naming → fetch variable definitions and class field names
```

### Post-failure mode (diagnosis — used by EFL)
Reads the actual Python error and maps it:
```python
ERROR_TO_CONTEXT = {
    "ImportError":       "imports",      # fetch all import statements
    "ModuleNotFoundError": "imports",
    "NameError":         "definitions",  # fetch all var/function definitions
    "UnboundLocalError": "definitions",
    "TypeError":         "signatures",   # fetch function signatures + type hints
    "AttributeError":    "signatures",
    "AssertionError":    "tests",        # fetch test cases + docstrings
    "ValueError":        "tests",
    "IndexError":        "tests",
    "KeyError":          "tests",
}
```

### Context extraction functions
```python
def fetch_imports(repo_files: dict) -> list[str]:
    """Extract all import lines from every file."""
    # Returns: ["from config import API_KEY, BASE_URL", "import requests", ...]

def fetch_definitions(repo_files: dict) -> list[str]:
    """Extract all function defs, class defs, and assignments."""
    # Returns: ["def fetch_weather(city: str) -> dict:", "API_KEY = 'abc123'", ...]

def fetch_signatures(repo_files: dict) -> list[str]:
    """Extract function signatures with docstrings and type hints."""
    # Returns: ["def fetch_weather(city: str) -> dict:\n    \"\"\"Fetch weather...\"\"\"", ...]

def fetch_tests(repo_files: dict) -> list[str]:
    """Extract test functions and assert statements."""
    # Returns: ["def test_weather():\n    assert wr.is_hot() == True", ...]
```

---

## Component 5: Prompt Builder

**Purpose:** Assemble the final prompt from selected context and the query.

**Input:**
- `selected_chunks: list[str]` — from HCCS scorer
- `router_extras: list[str]` — from type router
- `query: str` — the coding task
- `error_context: str` — (optional) previous error info from EFL

**Output:** A single prompt string sent to the code LLM
```
# Repository context (selected by HaluGuard):

# File: config.py
API_KEY = "abc123def456"
BASE_URL = "https://api.openweathermap.org/data/2.5"
CACHE_TTL = 300

# File: models.py
@dataclass
class WeatherReport:
    city: str
    temperature: float
    ...

# File: api.py
def fetch_weather(city: str) -> dict:
    ...

# Additional context:
# Variable definitions: CACHE_TTL = 300
# Field names: WeatherReport.temperature, WeatherReport.humidity

# Previous attempt failed with: ImportError (if EFL retry)

# Task: Write a function called get_forecast that takes a city name,
# fetches weather data using the existing API, and returns a
# WeatherReport object. Cache results for CACHE_TTL seconds.
```

---

## Component 6: Code Executor (Sandbox)

**Purpose:** Safely execute generated Python code and capture results.

**Input:**
- `code: str` — the generated Python code
- `test_code: str` — test assertions to verify correctness

**Output:** `ExecutionResult` dataclass
```python
@dataclass
class ExecutionResult:
    passed: bool                           # True if return code == 0
    stdout: str                            # captured stdout
    stderr: str                            # captured stderr
    error_type: str | None                 # "ImportError", "NameError", etc.
    error_message: str | None              # "cannot import name 'get_weather'"
    hallucination_type: HallucinationType | None  # RESOURCE, NAMING, MAPPING, LOGIC
```

**How it works:**
1. Write code + tests to a temporary .py file
2. Run `subprocess.run(["python", temp_file], capture_output=True, timeout=30)`
3. Parse stderr to extract error type and message
4. Map error type to hallucination category
5. Clean up temp file

**On Colab:** No Docker needed — use subprocess with timeout. Colab runs in an isolated VM already. For extra safety, use `resource.setrlimit()` to cap memory.

---

## Component 7: Execution Feedback Loop (EFL)

**Purpose:** Iteratively refine code by using execution errors to fetch better context.

**Input:**
- `query: str`
- `initial_context: list[str]` — from HCCS + router
- `test_code: str`
- `generate_fn` — function that calls the code LLM
- `retrieve_fn` — function that fetches context for a hallucination type

**Output:** Dict with final code, pass/fail, number of iterations, history
```python
{
    "code": "import time\nfrom api import fetch_weather...",
    "passed": True,
    "iterations": 2,
    "history": [
        {"iteration": 0, "passed": False, "error_type": "ImportError", "hallucination_type": "RESOURCE"},
        {"iteration": 1, "passed": True, "error_type": None}
    ]
}
```

**Algorithm:**
```
current_context = initial_context (from HCCS + router)

for iteration in 0, 1, 2:
    prompt = build_prompt(query, current_context, previous_error_if_any)
    code = generate_fn(prompt)
    result = execute(code, test_code)

    if result.passed:
        return {code, passed=True, iterations=iteration+1}

    # Diagnose failure
    hall_type = result.hallucination_type  # e.g., RESOURCE

    # Fetch targeted context
    new_context = retrieve_fn(query, hall_type)  # e.g., all import statements
    current_context += new_context  # add to existing, don't replace

return {best_code, passed=False, iterations=3}
```

---

## Data flow summary (weather app example)

```
Repo files (4 .py files)
    │
    ▼
Chunker → 4 chunks (one per file)
    │
    ▼
CodeBERT → 4 embeddings, each shape (768,), saved as chunks.npy
    │
Query: "Write get_forecast..."
    │
    ▼
CodeBERT → 1 query embedding, shape (768,)
    │
    ▼
HCCS Scorer: for each chunk, concat(query_emb, chunk_emb) → MLP → score
    → config.py: 0.94, models.py: 0.91, api.py: 0.89, tests: 0.52
    → Select top 3: [config.py, models.py, api.py]
    │
    ▼
Type Router: query_emb → classifier → naming=45% (highest)
    → Add: CACHE_TTL=300, WeatherReport field names
    │
    ▼
Prompt Builder: merge chunks + extras + query → prompt string (~1500 chars)
    │
    ▼
DeepSeek-Coder on T4: prompt → generated Python code (~30 lines)
    │
    ▼
Executor: run code + tests → PASSED (iteration 1)
    │
    ▼
Return: {code, passed=True, iterations=1}
```
