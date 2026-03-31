# HaluGuard Dry Run — Complete Worked Example

This document traces one concrete RepoBench example through every single step of the HaluGuard system, showing exact inputs and outputs at each stage.

## The RepoBench Example

A Python repository where the current file uses a function from another file:

### Current file being completed: `src/main.py`

```python
# import_statement:
from utils.data_loader import load_dataset

# cropped_code (code written so far):
import json
from utils.data_loader import load_dataset

def process_data(path: str):
    data = load_dataset(path)
    result = data.
```

### Context chunks (from other files in the repo):

```python
# context[0] — identifier: "DataLoader", path: "utils/data_loader.py"
class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load(self) -> dict:
        with open(self.filepath) as f:
            self.data = json.load(f)
        return self.data

    def filter_by(self, key: str, value: str) -> list:
        return [item for item in self.data if item.get(key) == value]

# context[1] — identifier: "load_dataset", path: "utils/data_loader.py"
def load_dataset(path: str) -> DataLoader:
    """Load a dataset from a JSON file and return a DataLoader instance."""
    loader = DataLoader(path)
    loader.load()
    return loader

# context[2] — identifier: "Config", path: "config/settings.py"
class Config:
    DATA_DIR = "/data"
    MAX_ITEMS = 1000
    VERBOSE = True

# context[3] — identifier: "test_load", path: "tests/test_data.py"
def test_load():
    loader = load_dataset("test_data.json")
    assert isinstance(loader, DataLoader)
    assert loader.data is not None
```

### Ground truth:
- `gold_snippet_index = 1` (the `load_dataset` function definition)
- `next_line = "result = data.filter_by('status', 'active')"`

## Step 1: Pre-compute Embeddings (offline)

CodeBERT encodes each text into a 768-dim vector:

```
query_emb = embed_code(cropped_code)        # shape (768,)
chunk_embs = [
    embed_code(context[0]["snippet"]),       # DataLoader class
    embed_code(context[1]["snippet"]),       # load_dataset function
    embed_code(context[2]["snippet"]),       # Config class
    embed_code(context[3]["snippet"]),       # test_load function
]                                            # shape (4, 768)
```

## Step 2: Generate Triplets (offline, from gold_snippet_index)

```python
# gold_snippet_index = 1, so context[1] is the positive
positive = context[1]["snippet"]  # load_dataset — returns DataLoader

# All others are negatives:
triplet_0 = (query=cropped_code, pos=context[1], neg=context[0])
triplet_1 = (query=cropped_code, pos=context[1], neg=context[2])
triplet_2 = (query=cropped_code, pos=context[1], neg=context[3])
```

## Step 3: HCCS Scoring (inference)

The trained MLP scores each chunk:

```
scores = scorer.score_chunks(query_emb, chunk_embs)
# Result: [0.72, 0.94, 0.31, 0.48]
#          DataLoader  load_dataset  Config  test_load
```

The gold chunk (index 1) gets the highest score because the MLP learned
that `load_dataset`'s definition is essential for predicting `data.filter_by()`.

## Step 4: Type Router (pre-emptive boosting)

Analyse `cropped_code`:
- Has method call `data.` → boost NAMING +0.15
- Has import `from utils.data_loader import` → boost RESOURCE +0.1

Classify each snippet:
- context[0]: has `class DataLoader` → NAMING
- context[1]: has `def load_dataset` → NAMING
- context[2]: has `class Config` → NAMING
- context[3]: path contains "test" → LOGIC

Apply boosts:
```
adjusted = [0.72 + 0.15,  0.94 + 0.15,  0.31 + 0.15,  0.48 + 0.0]
         = [0.87,          1.00,          0.46,          0.48]
```

Top-2 selected: indices [1, 0] → `load_dataset` and `DataLoader` definitions.

## Step 5: Build Prompt & Generate

```
# Cross-file context:
def load_dataset(path: str) -> DataLoader:
    """Load a dataset from a JSON file and return a DataLoader instance."""
    loader = DataLoader(path)
    loader.load()
    return loader

# Cross-file context:
class DataLoader:
    def __init__(self, filepath: str):
        ...
    def filter_by(self, key: str, value: str) -> list:
        return [item for item in self.data if item.get(key) == value]

from utils.data_loader import load_dataset

import json
from utils.data_loader import load_dataset

def process_data(path: str):
    data = load_dataset(path)
    result = data.
```

DeepSeek-Coder generates: `result = data.filter_by('status', 'active')`

## Step 6: Compare to Ground Truth

```
predicted:    "result = data.filter_by('status', 'active')"
ground_truth: "result = data.filter_by('status', 'active')"

Exact Match:     1.0  (identical!)
Edit Similarity: 1.0
```

## What would go wrong WITHOUT the gold context?

If the model only saw `Config` and `test_load` (no `DataLoader` definition):
- It wouldn't know `filter_by` exists as a method
- It might hallucinate: `result = data.get('status')` or `result = data['active']`
- These would fail with `AttributeError` (naming hallucination)

**This is the core insight:** Showing the model the function definition it needs
prevents it from guessing wrong method names and signatures.
