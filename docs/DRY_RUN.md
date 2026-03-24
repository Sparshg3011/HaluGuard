# HaluGuard Dry Run — Complete Worked Example

This document traces one concrete coding task through every single step of the HaluGuard system, showing exact inputs and outputs at each stage.

## The example repository

A weather app with 4 Python files:

### weather_app/api.py
```python
import requests
from config import API_KEY, BASE_URL

def fetch_weather(city: str) -> dict:
    """Fetch weather data for a city from OpenWeather API."""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(f"{BASE_URL}/weather", params=params)
    response.raise_for_status()
    return response.json()
```

### weather_app/config.py
```python
API_KEY = "abc123def456"
BASE_URL = "https://api.openweathermap.org/data/2.5"
DEFAULT_CITY = "Los Angeles"
CACHE_TTL = 300  # seconds
```

### weather_app/models.py
```python
from dataclasses import dataclass

@dataclass
class WeatherReport:
    city: str
    temperature: float
    humidity: int
    description: str

    def is_hot(self) -> bool:
        return self.temperature > 30.0

    def summary(self) -> str:
        return f"{self.city}: {self.temperature}C, {self.description}"
```

### weather_app/tests/test_models.py
```python
from models import WeatherReport

def test_weather_report():
    wr = WeatherReport("LA", 32.5, 60, "sunny")
    assert wr.is_hot() == True
    assert "LA" in wr.summary()
    assert "32.5" in wr.summary()
```

### The query
```
"Write a function called get_forecast that takes a city name,
fetches weather data using the existing API, and returns a
WeatherReport object. Cache results for CACHE_TTL seconds."
```

---

## Step 1: Chunking

**Input:** dict of 4 files
**Output:** list of 4 chunk strings

```python
chunks = [
    "# File: weather_app/api.py (lines 1-10)\nimport requests\nfrom config import API_KEY, BASE_URL\n\ndef fetch_weather(city: str) -> dict:\n    ...",
    "# File: weather_app/config.py (lines 1-4)\nAPI_KEY = \"abc123def456\"\nBASE_URL = \"https://api.openweathermap.org/data/2.5\"\nDEFAULT_CITY = \"Los Angeles\"\nCACHE_TTL = 300",
    "# File: weather_app/models.py (lines 1-15)\nfrom dataclasses import dataclass\n\n@dataclass\nclass WeatherReport:\n    city: str\n    temperature: float\n    ...",
    "# File: weather_app/tests/test_models.py (lines 1-8)\nfrom models import WeatherReport\n\ndef test_weather_report():\n    wr = WeatherReport(\"LA\", 32.5, 60, \"sunny\")\n    ..."
]
```

---

## Step 2: CodeBERT embedding

**Input:** 1 query string + 4 chunk strings
**Output:** 1 query vector (768,) + 4 chunk vectors (4, 768)

```python
# Each is actually 768 numbers. Showing simplified 4-number version:
query_emb  = [0.82, 0.45, 0.91, 0.33, ...]  # shape (768,)
chunk_embs = [
    [0.80, 0.50, 0.85, 0.30, ...],  # chunk 0: api.py
    [0.20, 0.10, 0.15, 0.90, ...],  # chunk 1: config.py
    [0.75, 0.40, 0.88, 0.25, ...],  # chunk 2: models.py
    [0.60, 0.35, 0.70, 0.20, ...],  # chunk 3: tests
]
```

Saved to disk: `np.save("data/embeddings/chunks.npy", chunk_embs)`

---

## Step 3: HCCS scoring

**Input:** query_emb (768,) + each chunk_emb (768,) → concatenated (1536,)
**Output:** one float per chunk

```python
# For each chunk: concat query + chunk → feed through MLP → get score
score_0 = mlp(concat(query_emb, chunk_embs[0]))  # api.py    → 0.89
score_1 = mlp(concat(query_emb, chunk_embs[1]))  # config.py → 0.94
score_2 = mlp(concat(query_emb, chunk_embs[2]))  # models.py → 0.91
score_3 = mlp(concat(query_emb, chunk_embs[3]))  # tests     → 0.52

# Sort descending, take top 3:
selected_indices = [1, 2, 0]  # config.py, models.py, api.py
selected_chunks = [chunks[1], chunks[2], chunks[0]]
```

**Key observation:** config.py ranks FIRST by prevention score despite having the LOWEST similarity to the query. This is the core insight of the paper — helpful ≠ similar.

---

## Step 4: Type routing

**Input:** query_emb (768,)
**Output:** probability distribution over 4 error types + extra context

```python
type_probs = type_classifier(query_emb)
# → {resource: 0.18, naming: 0.45, mapping: 0.30, logic: 0.07}
# Highest: naming (45%)

# Router fetches naming-prevention context:
router_extras = [
    "CACHE_TTL = 300",
    "DEFAULT_CITY = \"Los Angeles\"",
    "WeatherReport.city: str",
    "WeatherReport.temperature: float",  # NOT "temp"!
    "WeatherReport.humidity: int",
    "WeatherReport.description: str",
    "def fetch_weather(city: str) -> dict:",
]
```

---

## Step 5: Prompt assembly

**Input:** selected chunks + router extras + query
**Output:** one prompt string

```
# Repository context (selected by HaluGuard):

# File: config.py
API_KEY = "abc123def456"
BASE_URL = "https://api.openweathermap.org/data/2.5"
DEFAULT_CITY = "Los Angeles"
CACHE_TTL = 300

# File: models.py
from dataclasses import dataclass

@dataclass
class WeatherReport:
    city: str
    temperature: float
    humidity: int
    description: str
    def is_hot(self) -> bool: ...
    def summary(self) -> str: ...

# File: api.py
import requests
from config import API_KEY, BASE_URL

def fetch_weather(city: str) -> dict:
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(f"{BASE_URL}/weather", params=params)
    return response.json()

# Additional definitions:
# CACHE_TTL = 300
# WeatherReport fields: city (str), temperature (float), humidity (int), description (str)

# Task: Write a function called get_forecast that takes a city name,
# fetches weather data using the existing API, and returns a
# WeatherReport object. Cache results for CACHE_TTL seconds.
```

---

## Step 6: Code generation

**Input:** prompt string (above)
**Output:** generated Python code

```python
import time
from api import fetch_weather
from models import WeatherReport
from config import CACHE_TTL

_cache = {}

def get_forecast(city: str) -> WeatherReport:
    now = time.time()
    if city in _cache and (now - _cache[city][1]) < CACHE_TTL:
        return _cache[city][0]

    data = fetch_weather(city)
    report = WeatherReport(
        city=city,
        temperature=data["main"]["temp"],
        humidity=data["main"]["humidity"],
        description=data["weather"][0]["description"]
    )
    _cache[city] = (report, now)
    return report
```

---

## Step 7: Execution

**Input:** generated code + test code
**Output:** ExecutionResult

```python
# Test code:
test_code = """
from unittest.mock import patch
def test_get_forecast():
    mock_data = {"main": {"temp": 25.0, "humidity": 65}, "weather": [{"description": "clear sky"}]}
    with patch("api.fetch_weather", return_value=mock_data):
        result = get_forecast("London")
        assert isinstance(result, WeatherReport)
        assert result.temperature == 25.0
        assert result.city == "London"
"""

# Execution result:
ExecutionResult(
    passed=True,
    stdout="1 test passed",
    stderr="",
    error_type=None,
    hallucination_type=None
)
```

**EFL decision:** passed=True → return code. Done in 1 iteration.

---

## What if it failed? (EFL retry example)

Suppose the LLM had generated this instead:

```python
from api import get_weather       # WRONG! Function is called fetch_weather
from models import WeatherReport
```

### EFL iteration 1 result:
```python
ExecutionResult(
    passed=False,
    stderr="ImportError: cannot import name 'get_weather' from 'api'",
    error_type="ImportError",
    hallucination_type=HallucinationType.RESOURCE
)
```

### EFL fetches targeted context:
```python
# error_type = "ImportError" → category = "imports"
# Fetches all import-able names from the repo:
new_context = [
    "# Available in api.py: fetch_weather",
    "# Available in config.py: API_KEY, BASE_URL, DEFAULT_CITY, CACHE_TTL",
    "# Available in models.py: WeatherReport",
]
```

### EFL iteration 2 prompt includes:
- All original context
- Plus: "Previous attempt failed: ImportError: cannot import name 'get_weather'"
- Plus: "Available functions in api.py: fetch_weather"

### LLM generates corrected code:
```python
from api import fetch_weather     # FIXED!
from models import WeatherReport
```

### Execution → PASSED. Done in 2 iterations.

---

## How training data is created (for the HCCS scorer)

Using this same weather app example:

### Trial A: Include config.py + models.py as context
```
LLM generates → uses CACHE_TTL correctly, uses WeatherReport.temperature correctly
Execute → PASSED
```

### Trial B: Include only tests/test_models.py as context
```
LLM generates → invents CACHE_DURATION = 600, uses field name "temp" instead of "temperature"
Execute → FAILED (NameError: CACHE_DURATION not defined, or test assertion fails)
```

### Resulting triplet:
```json
{
    "query": "Write get_forecast that fetches weather and returns WeatherReport...",
    "positive_context": "# config.py\nAPI_KEY = ...\nCACHE_TTL = 300\n\n# models.py\n@dataclass\nclass WeatherReport: ...",
    "negative_context": "# tests/test_models.py\nfrom models import WeatherReport\ndef test_weather_report(): ...",
    "hallucination_type": 1,
    "task_id": "weather_001"
}
```

This triplet teaches the HCCS scorer: "For queries about writing functions that use config values and return dataclass objects, chunks containing the config file and the dataclass definition are more helpful than chunks containing only test code."

Multiply this by 3,000-5,000 such triplets across hundreds of different coding tasks, and the scorer learns general rules about what makes context helpful.
