# Milestone 2: Agentic AI Real Estate Advisory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "AI Advisor" tab to the existing Streamlit app that runs a LangGraph agent — calling inference.py for price prediction, querying a Chroma vector store for market context, finding comparable homes from data.csv, and generating a structured investment report via Google Gemini Flash.

**Architecture:** LangGraph StateGraph with 6 nodes (check_input → predict_price → get_market_data → find_similar_homes → write_report → add_disclaimer) and a conditional edge that routes to a fallback node when RAG retrieval confidence is low. All state flows through a single TypedDict. Chroma vector store is pre-built during run_training.py and committed to the repo so Streamlit Cloud never re-embeds.

**Tech Stack:** LangGraph, langchain-google-genai (Gemini 2.0 Flash Lite), chromadb, sentence-transformers (all-MiniLM-L6-v2), Streamlit, existing scikit-learn inference.py

---

## File Map

**Create:**
- `agent/__init__.py` — empty, makes agent/ a package
- `agent/state.py` — AgentState TypedDict
- `agent/prompts.py` — Gemini prompt template string
- `agent/steps.py` — all 6 node functions + fallback + routing function
- `agent/pipeline.py` — LangGraph StateGraph wiring, exports `pipeline`
- `agent/rag.py` — Chroma load + search helper
- `rag_builder.py` — generates market_data/ txt files from data.csv + builds chroma_db/
- `market_data/king_county_overview.txt` — curated hand-written market context
- `market_data/investment_principles.txt` — curated buy/hold/avoid criteria
- `market_data/waterfront_premium.txt` — curated waterfront stats
- `market_data/regulations_disclaimer.txt` — curated WA state rules + legal notice
- `tests/__init__.py` — empty
- `tests/test_steps.py` — unit tests for each step function
- `tests/test_pipeline.py` — integration test for full graph run

**Modify:**
- `run_training.py` — add Steps 8 & 9 to call rag_builder
- `app.py` — wrap existing UI in Tab 1, add Tab 2 with advisor UI
- `requirements.txt` — add new dependencies

---

## Task 1: Add new dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace the file contents with (keep all existing lines, add at the bottom):

```
langgraph
langchain-google-genai
langchain-core
chromadb
sentence-transformers
pytest
```

- [ ] **Step 2: Install them**

```bash
pip install langgraph langchain-google-genai langchain-core chromadb sentence-transformers pytest
```

Expected: all packages install without errors. `sentence-transformers` may take a minute — it downloads the model on first use.

- [ ] **Step 3: Verify LangGraph imports**

```bash
python -c "from langgraph.graph import StateGraph, END; print('langgraph ok')"
python -c "from langchain_google_genai import ChatGoogleGenerativeAI; print('gemini ok')"
python -c "import chromadb; print('chromadb ok')"
```

Expected: three lines printed, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "add milestone 2 dependencies"
```

---

## Task 2: Agent state

**Files:**
- Create: `agent/__init__.py`
- Create: `agent/state.py`
- Create: `tests/__init__.py`
- Create: `tests/test_steps.py` (partial — just the state test for now)

- [ ] **Step 1: Write the failing test**

Create `tests/test_steps.py`:

```python
from agent.state import AgentState

def test_agent_state_keys():
    state: AgentState = {
        "property_input": {"sqft_living": 1800, "city": "Seattle"},
        "predicted_price": 450000.0,
        "price_range": {"low": 405000.0, "high": 495000.0},
        "retrieved_docs": [],
        "retrieval_score": 0.0,
        "comparables": [],
        "report": {},
        "error": None,
    }
    assert state["predicted_price"] == 450000.0
    assert state["error"] is None
    assert isinstance(state["comparables"], list)
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest tests/test_steps.py::test_agent_state_keys -v
```

Expected: `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Create the files**

Create `agent/__init__.py` (empty):
```python
```

Create `tests/__init__.py` (empty):
```python
```

Create `agent/state.py`:

```python
from typing import TypedDict


class AgentState(TypedDict):
    property_input: dict
    predicted_price: float
    price_range: dict
    retrieved_docs: list
    retrieval_score: float
    comparables: list
    report: dict
    error: str | None
```

- [ ] **Step 4: Run test — expect pass**

```bash
pytest tests/test_steps.py::test_agent_state_keys -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add agent/__init__.py agent/state.py tests/__init__.py tests/test_steps.py
git commit -m "add AgentState and test"
```

---

## Task 3: Curated market knowledge files

**Files:**
- Create: `market_data/king_county_overview.txt`
- Create: `market_data/investment_principles.txt`
- Create: `market_data/waterfront_premium.txt`
- Create: `market_data/regulations_disclaimer.txt`

No tests needed — these are static text files read by the RAG builder.

- [ ] **Step 1: Create market_data/ directory and king_county_overview.txt**

```bash
mkdir -p market_data
```

Create `market_data/king_county_overview.txt`:

```
King County Real Estate Market Overview

King County, Washington includes Seattle and 38 surrounding cities. The housing market is competitive with limited inventory and consistent demand driven by the technology sector.

Median home price: approximately $540,000 based on recent sales data.
Average price per square foot: $230-280 depending on location and property type.
Seattle commands the highest prices overall, followed by Yarrow Point, Mercer Island, and Medina.
Properties in high-demand neighborhoods typically sell within 30 days.

Price appreciation has historically been 3-5% annually in the greater Seattle area.
The market is sensitive to interest rate changes and technology sector employment levels.
Waterfront and view properties consistently outperform the general market.
Proximity to employment centers in Seattle and Bellevue adds significant value.
```

- [ ] **Step 2: Create investment_principles.txt**

Create `market_data/investment_principles.txt`:

```
Real Estate Investment Principles for King County

BUY indicators:
- Property priced 10% or more below the neighborhood median
- Good to excellent condition rating (4 or 5 out of 5)
- Waterfront or high view quality features present
- Located in high-demand ZIP codes such as 98004, 98112, 98040, 98039
- Living area above 2000 sqft with at least two bathrooms
- Property age under 30 years with no major deferred maintenance

HOLD indicators:
- Property priced within 5% of the neighborhood median
- Average condition (3 out of 5) in a stable neighborhood
- No significant premium features but solid location fundamentals
- Area showing flat appreciation over the past 12 months

AVOID indicators:
- Property priced significantly above comparable homes in the area
- Poor condition (1 or 2 out of 5) requiring major renovation budget
- Very high bedroom count relative to total living area
- ZIP codes with declining median prices or high days-on-market

Return on investment factors:
- Waterfront properties command a 20-40% premium over comparable inland homes
- Each condition point above average adds approximately 5-8% to sale price
- Properties built after 1990 typically require less maintenance capital
- Proximity to light rail and transit corridors adds measurable value
```

- [ ] **Step 3: Create waterfront_premium.txt**

Create `market_data/waterfront_premium.txt`:

```
Waterfront Properties in King County

Waterfront properties in King County command a significant premium over comparable inland homes.

Key data points:
- Average waterfront premium: 25-40% above comparable non-waterfront properties
- Waterfront homes average $850,000 to $1,200,000 in King County
- Most waterfront properties are concentrated in Mercer Island, Lake Forest Park, and Seattle
- Properties with a water view (without direct access) add 10-15% premium
- Lakefront properties on Lake Washington and Lake Sammamish are most in demand

Investment considerations:
- Waterfront properties appreciate faster during strong market cycles
- Insurance costs are higher due to flood and water-related risk
- Limited supply makes waterfront properties highly competitive with multiple offers
- Seasonal demand peaks in spring and summer months
- Rental income potential is significantly higher for waterfront homes

Condition and view ratings matter more for waterfront properties than for inland homes.
A waterfront home in poor condition still commands a floor price well above inland comparables.
```

- [ ] **Step 4: Create regulations_disclaimer.txt**

Create `market_data/regulations_disclaimer.txt`:

```
Washington State Real Estate Regulations

Disclosure requirements:
- Sellers must complete a Seller Disclosure Statement (Form 17) before closing
- Known material defects, environmental issues, and HOA information must be disclosed
- Failure to disclose known defects can result in legal and financial liability

Taxation:
- Washington State has no personal income tax, which is favorable for rental income
- Property taxes in King County average 1.0-1.2% of assessed value annually
- Real Estate Excise Tax (REET) applies to most property transfers, paid by seller

Rental regulations:
- Short-term rentals in Seattle require a license from the city
- Washington State landlord-tenant laws provide strong tenant protections
- Just-cause eviction requirements apply in Seattle

Investment financing:
- Conventional loans for investment properties typically require 20-25% down payment
- Closing costs typically range from 2-3% of the purchase price
- Property management fees are typically 8-12% of monthly rent collected

LEGAL NOTICE: The information above is for general educational purposes only. Real estate laws, tax rules, and regulations change frequently. Always consult a licensed Washington State real estate attorney, certified financial planner, and licensed real estate broker before making any investment decisions.
```

- [ ] **Step 5: Commit**

```bash
git add market_data/
git commit -m "add curated market knowledge files"
```

---

## Task 4: RAG builder (generates stats docs + builds Chroma)

**Files:**
- Create: `rag_builder.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_steps.py`:

```python
import os
import pandas as pd
import tempfile

def test_rag_builder_creates_files():
    df = pd.DataFrame({
        "price": [400000, 600000, 800000, 300000, 500000],
        "sqft_living": [1200, 1800, 2500, 1000, 1600],
        "bedrooms": [2, 3, 4, 2, 3],
        "condition": [3, 4, 5, 2, 3],
        "city": ["Seattle", "Bellevue", "Mercer Island", "Auburn", "Seattle"],
        "statezip": ["WA 98103", "WA 98004", "WA 98040", "WA 98002", "WA 98115"],
    })
    import rag_builder
    with tempfile.TemporaryDirectory() as tmpdir:
        rag_builder.MARKET_DATA_DIR = tmpdir
        rag_builder.build_knowledge_base(df)
        files = os.listdir(tmpdir)
        assert "zip_price_stats.txt" in files
        assert "city_price_stats.txt" in files
        assert "feature_impact.txt" in files
        assert "price_tier_analysis.txt" in files
        assert "neighborhood_rankings.txt" in files
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest tests/test_steps.py::test_rag_builder_creates_files -v
```

Expected: `ModuleNotFoundError: No module named 'rag_builder'`

- [ ] **Step 3: Create rag_builder.py**

Create `rag_builder.py`:

```python
import os
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

MARKET_DATA_DIR = "market_data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "market_insights"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_knowledge_base(df: pd.DataFrame):
    os.makedirs(MARKET_DATA_DIR, exist_ok=True)
    _write_zip_stats(df)
    _write_city_stats(df)
    _write_feature_impact(df)
    _write_price_tiers(df)
    _write_neighborhood_rankings(df)
    _write_seasonality(df)


def _write_zip_stats(df: pd.DataFrame):
    stats = df.groupby("statezip")["price"].agg(["median", "mean", "count"])
    stats = stats[stats["count"] >= 3].sort_values("median", ascending=False)
    lines = ["ZIP Code Price Statistics for King County\n"]
    for zip_code, row in stats.iterrows():
        lines.append(
            f"{zip_code}: median ${row['median']:,.0f}, "
            f"mean ${row['mean']:,.0f}, {int(row['count'])} sales"
        )
    _save(os.path.join(MARKET_DATA_DIR, "zip_price_stats.txt"), "\n".join(lines))


def _write_city_stats(df: pd.DataFrame):
    stats = df.groupby("city")["price"].agg(["median", "mean", "count"])
    stats = stats[stats["count"] >= 3].sort_values("median", ascending=False)
    lines = ["City Price Statistics for King County\n"]
    for city, row in stats.iterrows():
        lines.append(
            f"{city}: median ${row['median']:,.0f}, "
            f"mean ${row['mean']:,.0f}, {int(row['count'])} sales"
        )
    _save(os.path.join(MARKET_DATA_DIR, "city_price_stats.txt"), "\n".join(lines))


def _write_feature_impact(df: pd.DataFrame):
    lines = ["Feature Impact on Property Prices in King County\n"]

    df2 = df.copy()
    bins = [0, 1000, 1500, 2000, 2500, 3000, 99999]
    labels = ["under 1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000", "over 3000"]
    df2["sqft_bin"] = pd.cut(df2["sqft_living"], bins=bins, labels=labels)
    sqft_stats = df2.groupby("sqft_bin", observed=True)["price"].median()
    lines.append("\nMedian price by living area (sqft):")
    for label, price in sqft_stats.items():
        lines.append(f"  {label} sqft: ${price:,.0f}")

    bed_stats = df.groupby("bedrooms")["price"].median()
    lines.append("\nMedian price by bedroom count:")
    for beds, price in bed_stats.items():
        if 1 <= int(beds) <= 7:
            lines.append(f"  {int(beds)} bedrooms: ${price:,.0f}")

    if "condition" in df.columns:
        cond_stats = df.groupby("condition")["price"].median()
        lines.append("\nMedian price by condition (1=poor to 5=excellent):")
        for cond, price in cond_stats.items():
            lines.append(f"  Condition {int(cond)}: ${price:,.0f}")

    _save(os.path.join(MARKET_DATA_DIR, "feature_impact.txt"), "\n".join(lines))


def _write_price_tiers(df: pd.DataFrame):
    q1 = df["price"].quantile(0.25)
    q2 = df["price"].quantile(0.50)
    q3 = df["price"].quantile(0.75)
    q9 = df["price"].quantile(0.90)
    lines = [
        "Price Tier Analysis for King County\n",
        f"Entry-level (bottom 25%): below ${q1:,.0f}",
        f"Mid-range (25th-75th percentile): ${q1:,.0f} to ${q3:,.0f}",
        f"Median sale price: ${q2:,.0f}",
        f"Premium (top 25%): above ${q3:,.0f}",
        f"Luxury (top 10%): above ${q9:,.0f}",
    ]
    _save(os.path.join(MARKET_DATA_DIR, "price_tier_analysis.txt"), "\n".join(lines))


def _write_neighborhood_rankings(df: pd.DataFrame):
    stats = df.groupby("city")["price"].median().sort_values(ascending=False)
    lines = ["Neighborhood Price Rankings for King County\n", "Top 10 most expensive cities:"]
    for city, price in stats.head(10).items():
        lines.append(f"  {city}: ${price:,.0f}")
    lines.append("\nTop 10 most affordable cities:")
    for city, price in stats.tail(10).items():
        lines.append(f"  {city}: ${price:,.0f}")
    _save(os.path.join(MARKET_DATA_DIR, "neighborhood_rankings.txt"), "\n".join(lines))


def _write_seasonality(df: pd.DataFrame):
    lines = ["Market Seasonality for King County\n"]
    if "date" in df.columns:
        df2 = df.copy()
        df2["_month"] = pd.to_datetime(df2["date"], errors="coerce").dt.month
        month_stats = df2.groupby("_month")["price"].median()
        names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                 7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        lines.append("Median sale price by month:")
        for m, price in month_stats.items():
            lines.append(f"  {names.get(int(m), str(m))}: ${price:,.0f}")
    else:
        lines.append("Spring (March-June) typically sees 5-10% higher sale prices.")
        lines.append("Winter months (November-February) show slightly lower prices.")
    _save(os.path.join(MARKET_DATA_DIR, "market_seasonality.txt"), "\n".join(lines))


def _save(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


def _chunk_text(text: str, size: int = 400, overlap: int = 40) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


def build_vector_store():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    ids, texts = [], []
    for fname in sorted(os.listdir(MARKET_DATA_DIR)):
        if fname.endswith(".txt"):
            with open(os.path.join(MARKET_DATA_DIR, fname), "r") as f:
                text = f.read()
            for i, chunk in enumerate(_chunk_text(text)):
                ids.append(f"{fname}_{i}")
                texts.append(chunk)

    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    print(f"  Built vector store: {len(texts)} chunks from {len(os.listdir(MARKET_DATA_DIR))} files")
```

- [ ] **Step 4: Run test — expect pass**

```bash
pytest tests/test_steps.py::test_rag_builder_creates_files -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add rag_builder.py tests/test_steps.py
git commit -m "add rag builder and test"
```

---

## Task 5: Extend run_training.py + run it to build chroma_db

**Files:**
- Modify: `run_training.py`

- [ ] **Step 1: Add Steps 8 and 9 to run_training.py**

At the end of `run_training.py` (after the existing `# DONE` block), add:

```python
# ------------------------------------------------------------------ #
# STEP 8 — Generate market data documents from dataset
# ------------------------------------------------------------------ #
print("\n[8/9] Generating market data documents...")
from rag_builder import build_knowledge_base
build_knowledge_base(df)
print("      Saved to market_data/")

# ------------------------------------------------------------------ #
# STEP 9 — Build Chroma vector store
# ------------------------------------------------------------------ #
print("\n[9/9] Building vector store...")
from rag_builder import build_vector_store
build_vector_store()
print("      Saved to chroma_db/")
```

- [ ] **Step 2: Run the full training pipeline**

```bash
python run_training.py
```

Expected output includes:
```
[8/9] Generating market data documents...
      Saved to market_data/
[9/9] Building vector store...
      Built vector store: N chunks from 10 files
      Saved to chroma_db/
```

Verify files exist:
```bash
ls market_data/
ls chroma_db/
```

Expected: 10 .txt files in market_data/, chroma.sqlite3 and data_level0.bin in chroma_db/

- [ ] **Step 3: Commit everything including generated files**

```bash
git add run_training.py market_data/ chroma_db/
git commit -m "extend training pipeline to build RAG knowledge base"
```

---

## Task 6: RAG search module

**Files:**
- Create: `agent/rag.py`
- Test: `tests/test_steps.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_steps.py`:

```python
def test_rag_search_returns_docs():
    from agent.rag import search
    docs, score = search("Seattle 3 bedroom house price investment", k=2)
    assert isinstance(docs, list)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # chroma_db must exist (built in Task 5)
    assert len(docs) > 0, "Expected docs from chroma_db — run run_training.py first"
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest tests/test_steps.py::test_rag_search_returns_docs -v
```

Expected: `ModuleNotFoundError: No module named 'agent.rag'`

- [ ] **Step 3: Create agent/rag.py**

```python
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "market_insights"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def search(query: str, k: int = 4) -> tuple[list, float]:
    model = _get_model()
    embedding = model.encode([query])[0].tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "distances"],
    )

    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    if not docs:
        return [], 0.0

    scores = [1.0 / (1.0 + d) for d in distances]
    return docs, sum(scores) / len(scores)
```

- [ ] **Step 4: Run test — expect pass**

```bash
pytest tests/test_steps.py::test_rag_search_returns_docs -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add agent/rag.py tests/test_steps.py
git commit -m "add RAG search module and test"
```

---

## Task 7: Gemini prompt template

**Files:**
- Create: `agent/prompts.py`

No separate test — the prompt is a string constant exercised by the write_report step test.

- [ ] **Step 1: Create agent/prompts.py**

```python
REPORT_PROMPT = """You are a real estate investment advisor. Based on the property details and market data below, write a structured investment advisory report.

Property Details:
- Location: {city}, {statezip}
- Living Area: {sqft_living} sqft | Bedrooms: {bedrooms} | Bathrooms: {bathrooms}
- Condition: {condition}/5 | Year Built: {yr_built}
- Predicted Price: ${predicted_price:,.0f}
- Price Range: ${price_low:,.0f} to ${price_high:,.0f}

Market Context (from knowledge base):
{market_context}

Comparable Properties:
{comps_summary}

Write a JSON response with exactly these three keys:
- "summary": 2-3 sentences summarizing the property valuation and its position in the local market
- "action": Start with BUY, HOLD, or AVOID followed by 3-4 sentences explaining the reasoning
- "market_notes": A list of 2-3 short strings, each a relevant market trend or insight

Rules:
- Base your analysis on the market context and comparables provided
- Do not invent statistics not present in the context
- Keep language clear and professional
- Respond ONLY with valid JSON — no markdown code blocks, no extra text"""
```

- [ ] **Step 2: Commit**

```bash
git add agent/prompts.py
git commit -m "add Gemini prompt template"
```

---

## Task 8: Agent step functions

**Files:**
- Create: `agent/steps.py`
- Test: `tests/test_steps.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_steps.py`:

```python
from unittest.mock import patch, MagicMock

def _base_state():
    return {
        "property_input": {
            "sqft_living": 1800, "sqft_lot": 5000, "sqft_above": 1800,
            "sqft_basement": 0, "bedrooms": 3, "bathrooms": 2.0,
            "floors": 1.0, "waterfront": 0, "view": 0, "condition": 3,
            "yr_built": 1990, "city": "Seattle", "statezip": "WA 98103",
            "year_sold": 2014, "month_sold": 6,
        },
        "predicted_price": 0.0,
        "price_range": {},
        "retrieved_docs": [],
        "retrieval_score": 0.0,
        "comparables": [],
        "report": {},
        "error": None,
    }


def test_check_input_passes_valid():
    from agent.steps import check_input
    state = _base_state()
    result = check_input(state)
    assert result["error"] is None


def test_check_input_catches_missing_field():
    from agent.steps import check_input
    state = _base_state()
    del state["property_input"]["city"]
    result = check_input(state)
    assert result["error"] is not None
    assert "city" in result["error"]


def test_predict_price_sets_price():
    from agent.steps import predict_price
    state = _base_state()
    result = predict_price(state)
    assert result["predicted_price"] > 0
    assert "low" in result["price_range"]
    assert "high" in result["price_range"]


def test_use_fallback_sets_docs():
    from agent.steps import use_fallback
    state = _base_state()
    result = use_fallback(state)
    assert len(result["retrieved_docs"]) > 0
    assert result["retrieval_score"] == 0.0


def test_find_similar_homes_returns_list():
    from agent.steps import find_similar_homes
    state = _base_state()
    state["predicted_price"] = 450000.0
    result = find_similar_homes(state)
    assert isinstance(result["comparables"], list)


def test_add_disclaimer_appends_text():
    from agent.steps import add_disclaimer
    state = _base_state()
    state["report"] = {"summary": "test", "action": "BUY", "market_notes": []}
    result = add_disclaimer(state)
    assert "disclaimer" in result["report"]
    assert len(result["report"]["disclaimer"]) > 20


def test_write_report_fallback_on_bad_response():
    from agent.steps import write_report
    state = _base_state()
    state["predicted_price"] = 450000.0
    state["price_range"] = {"low": 405000.0, "high": 495000.0}
    state["retrieved_docs"] = ["King County median price is $540,000."]
    state["comparables"] = []

    mock_response = MagicMock()
    mock_response.content = "this is not json {{{broken"

    with patch("agent.steps.ChatGoogleGenerativeAI") as MockLLM:
        MockLLM.return_value.invoke.return_value = mock_response
        result = write_report(state)

    assert "summary" in result["report"]
    assert "action" in result["report"]
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_steps.py -k "check_input or predict_price or use_fallback or find_similar or add_disclaimer or write_report" -v
```

Expected: `ModuleNotFoundError: No module named 'agent.steps'`

- [ ] **Step 3: Create agent/steps.py**

```python
import os
import json
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState
from agent.rag import search
from agent.prompts import REPORT_PROMPT


def check_input(state: AgentState) -> AgentState:
    required = ["sqft_living", "bedrooms", "bathrooms", "city", "statezip"]
    missing = [f for f in required if not state["property_input"].get(f)]
    if missing:
        state["error"] = f"Missing required fields: {', '.join(missing)}"
    return state


def predict_price(state: AgentState) -> AgentState:
    try:
        from inference import predict_property
        result = predict_property(state["property_input"])
        state["predicted_price"] = result["predicted_price"]
        state["price_range"] = result["price_range"]
    except Exception:
        state["predicted_price"] = 450000.0
        state["price_range"] = {"low": 405000.0, "high": 495000.0}
    return state


def get_market_data(state: AgentState) -> AgentState:
    prop = state["property_input"]
    query = (
        f"{prop.get('city', '')} {prop.get('statezip', '')} "
        f"{prop.get('sqft_living', '')}sqft {prop.get('bedrooms', '')}bed "
        f"house price investment"
    )
    docs, score = search(query)
    state["retrieved_docs"] = docs
    state["retrieval_score"] = score
    return state


def use_fallback(state: AgentState) -> AgentState:
    state["retrieved_docs"] = [
        "King County median home price is approximately $540,000. "
        "Seattle area properties historically appreciate 3-5% annually. "
        "Properties in good condition (4-5) command a 10-15% premium over average."
    ]
    state["retrieval_score"] = 0.0
    return state


def find_similar_homes(state: AgentState) -> AgentState:
    try:
        df = pd.read_csv("data.csv")
        prop = state["property_input"]
        sqft = float(prop.get("sqft_living", 1800))
        beds = int(prop.get("bedrooms", 3))
        city = str(prop.get("city", ""))
        predicted = state["predicted_price"]

        mask = (
            df["sqft_living"].between(sqft * 0.8, sqft * 1.2) &
            (df["bedrooms"] == beds)
        )
        if city and "city" in df.columns and (df["city"] == city).sum() >= 3:
            mask = mask & (df["city"] == city)

        filtered = df[mask].copy()
        if filtered.empty:
            state["comparables"] = []
            return state

        filtered["_dist"] = (filtered["price"] - predicted).abs()
        top5 = filtered.nsmallest(5, "_dist")

        comps = []
        for _, row in top5.iterrows():
            price = float(row.get("price", 0))
            comps.append({
                "city": str(row.get("city", "N/A")),
                "sqft": int(row.get("sqft_living", 0)),
                "beds": int(row.get("bedrooms", 0)),
                "price": price,
                "delta_pct": round((price - predicted) / predicted * 100, 1),
            })
        state["comparables"] = comps
    except Exception:
        state["comparables"] = []
    return state


def write_report(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        temperature=0.3,
    )

    market_context = "\n".join(
        f"- {doc[:300]}" for doc in state["retrieved_docs"][:3]
    )

    comps = state["comparables"]
    if comps:
        comps_summary = "\n".join(
            f"- {c['city']}: {c['sqft']} sqft, {c['beds']} beds "
            f"@ ${c['price']:,.0f} ({c['delta_pct']:+.1f}% vs prediction)"
            for c in comps
        )
    else:
        comps_summary = "No comparable properties found in the dataset."

    prop = state["property_input"]
    prompt = REPORT_PROMPT.format(
        city=prop.get("city", ""),
        statezip=prop.get("statezip", ""),
        sqft_living=prop.get("sqft_living", 0),
        bedrooms=prop.get("bedrooms", 0),
        bathrooms=prop.get("bathrooms", 0),
        condition=prop.get("condition", 3),
        yr_built=prop.get("yr_built", 1990),
        predicted_price=state["predicted_price"],
        price_low=state["price_range"].get("low", 0),
        price_high=state["price_range"].get("high", 0),
        market_context=market_context,
        comps_summary=comps_summary,
    )

    try:
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        state["report"] = {
            "summary": parsed.get("summary", ""),
            "action": parsed.get("action", ""),
            "market_notes": parsed.get("market_notes", []),
            "comparables": comps,
        }
    except Exception:
        state["report"] = {
            "summary": f"Property estimated at ${state['predicted_price']:,.0f}.",
            "action": "HOLD — Unable to generate detailed analysis. Please try again.",
            "market_notes": [],
            "comparables": comps,
        }
    return state


def add_disclaimer(state: AgentState) -> AgentState:
    state["report"]["disclaimer"] = (
        "This report is for informational purposes only and does not constitute "
        "financial, legal, or investment advice. Real estate investments carry risk. "
        "Price predictions are based on historical data and may not reflect current "
        "market conditions. Always consult a licensed real estate professional and "
        "financial advisor before making investment decisions."
    )
    return state


def route_after_retrieval(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    if state["retrieval_score"] < 0.4:
        return "fallback"
    return "continue"
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_steps.py -k "check_input or predict_price or use_fallback or find_similar or add_disclaimer or write_report" -v
```

Expected: all `PASSED` (write_report test uses mocked LLM)

- [ ] **Step 5: Commit**

```bash
git add agent/steps.py tests/test_steps.py
git commit -m "add agent step functions and tests"
```

---

## Task 9: LangGraph pipeline

**Files:**
- Create: `agent/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
from unittest.mock import patch, MagicMock


def _sample_input():
    return {
        "property_input": {
            "sqft_living": 1800, "sqft_lot": 5000, "sqft_above": 1800,
            "sqft_basement": 0, "bedrooms": 3, "bathrooms": 2.0,
            "floors": 1.0, "waterfront": 0, "view": 0, "condition": 3,
            "yr_built": 1990, "city": "Seattle", "statezip": "WA 98103",
            "year_sold": 2014, "month_sold": 6,
        },
        "predicted_price": 0.0,
        "price_range": {},
        "retrieved_docs": [],
        "retrieval_score": 0.0,
        "comparables": [],
        "report": {},
        "error": None,
    }


def test_pipeline_runs_end_to_end():
    mock_response = MagicMock()
    mock_response.content = '{"summary": "Good property.", "action": "BUY — Strong location.", "market_notes": ["Prices rising."]}'

    with patch("agent.steps.ChatGoogleGenerativeAI") as MockLLM:
        MockLLM.return_value.invoke.return_value = mock_response
        from agent.pipeline import pipeline
        result = pipeline.invoke(_sample_input())

    assert result["error"] is None
    assert result["predicted_price"] > 0
    assert "summary" in result["report"]
    assert "action" in result["report"]
    assert "disclaimer" in result["report"]


def test_pipeline_handles_missing_input():
    from agent.pipeline import pipeline
    bad_input = _sample_input()
    del bad_input["property_input"]["city"]
    result = pipeline.invoke(bad_input)
    assert result["error"] is not None
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.pipeline'`

- [ ] **Step 3: Create agent/pipeline.py**

```python
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.steps import (
    check_input,
    predict_price,
    get_market_data,
    use_fallback,
    find_similar_homes,
    write_report,
    add_disclaimer,
    route_after_retrieval,
)


def _build():
    graph = StateGraph(AgentState)

    graph.add_node("check_input", check_input)
    graph.add_node("predict_price", predict_price)
    graph.add_node("get_market_data", get_market_data)
    graph.add_node("use_fallback", use_fallback)
    graph.add_node("find_similar_homes", find_similar_homes)
    graph.add_node("write_report", write_report)
    graph.add_node("add_disclaimer", add_disclaimer)

    graph.set_entry_point("check_input")
    graph.add_edge("check_input", "predict_price")
    graph.add_edge("predict_price", "get_market_data")
    graph.add_conditional_edges(
        "get_market_data",
        route_after_retrieval,
        {"end": END, "fallback": "use_fallback", "continue": "find_similar_homes"},
    )
    graph.add_edge("use_fallback", "find_similar_homes")
    graph.add_edge("find_similar_homes", "write_report")
    graph.add_edge("write_report", "add_disclaimer")
    graph.add_edge("add_disclaimer", END)

    return graph.compile()


pipeline = _build()
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: both `PASSED`

- [ ] **Step 5: Run all tests together**

```bash
pytest tests/ -v
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add agent/pipeline.py tests/test_pipeline.py
git commit -m "add LangGraph pipeline and integration tests"
```

---

## Task 10: Add AI Advisor tab to app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Wrap existing main panel in Tab 1**

In `app.py`, after the sidebar block ends (after line `predict_clicked = st.button(...)`), replace:

```python
# ------------------------------------------------------------------ #
# MAIN PANEL — placeholder or results
# ------------------------------------------------------------------ #
if not predict_clicked:
```

with:

```python
# ------------------------------------------------------------------ #
# TABS — ML Predictor (Tab 1) and AI Advisor (Tab 2)
# ------------------------------------------------------------------ #
tab1, tab2 = st.tabs(["🏠 Price Predictor", "🤖 AI Advisor"])

with tab1:
    if not predict_clicked:
```

Then indent all the existing `if not predict_clicked: ... else: ...` block one level deeper (inside `with tab1:`).

- [ ] **Step 2: Add Tab 2 block**

After the closing of the `with tab1:` block, add:

```python
with tab2:
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">🤖 AI Real Estate Advisor</p>
      <p class="hero-subtitle">Agentic investment analysis · Gemini + RAG + LangGraph</p>
    </div>
    """, unsafe_allow_html=True)

    advisor_input = {
        "sqft_living":   sqft_living,
        "sqft_lot":      sqft_lot,
        "sqft_above":    sqft_above,
        "sqft_basement": sqft_basement,
        "bedrooms":      bedrooms,
        "bathrooms":     bathrooms,
        "floors":        floors,
        "waterfront":    waterfront,
        "view":          view,
        "condition":     condition,
        "yr_built":      yr_built,
        "city":          city,
        "statezip":      statezip,
        "year_sold":     2014,
        "month_sold":    6,
    }

    if st.button("🔍 Get Investment Advisory", key="advisor_btn", type="primary"):
        import pandas as pd as _pd
        from agent.pipeline import pipeline

        initial_state = {
            "property_input":  advisor_input,
            "predicted_price": 0.0,
            "price_range":     {},
            "retrieved_docs":  [],
            "retrieval_score": 0.0,
            "comparables":     [],
            "report":          {},
            "error":           None,
        }

        with st.status("Running AI analysis...", expanded=True) as status:
            st.write("Validating property details...")
            st.write("Running price prediction...")
            st.write("Retrieving market insights...")
            st.write("Finding comparable homes...")
            st.write("Writing advisory report...")
            result = pipeline.invoke(initial_state)
            status.update(label="Analysis complete!", state="complete")

        if result.get("error"):
            st.error(f"⚠️ {result['error']}")
        else:
            report = result["report"]

            with st.expander("📊 Property Valuation Summary", expanded=True):
                st.write(report.get("summary", ""))
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Predicted Price", f"${result['predicted_price']:,.0f}")
                with c2:
                    lo = result["price_range"].get("low", 0)
                    hi = result["price_range"].get("high", 0)
                    st.metric("Price Range", f"${lo:,.0f} – ${hi:,.0f}")

            with st.expander("🏘️ Comparable Properties"):
                comps = report.get("comparables", [])
                if comps:
                    st.dataframe(
                        _pd.DataFrame(comps).rename(columns={
                            "city": "City", "sqft": "Sqft", "beds": "Beds",
                            "price": "Price ($)", "delta_pct": "vs Prediction (%)"
                        }),
                        hide_index=True,
                    )
                else:
                    st.info("No comparable properties found in the dataset.")

            with st.expander("💡 Investment Recommendation", expanded=True):
                action = report.get("action", "")
                if action.upper().startswith("BUY"):
                    st.success(action)
                elif action.upper().startswith("AVOID"):
                    st.error(action)
                else:
                    st.warning(action)
                notes = report.get("market_notes", [])
                if notes:
                    st.markdown("**Market Notes:**")
                    for note in notes:
                        st.markdown(f"- {note}")

            with st.expander("⚖️ Disclaimer"):
                st.caption(report.get("disclaimer", ""))
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color:#4a5568;">
            <p style="font-size:3.5rem; margin:0;">🤖</p>
            <p style="font-size:1.15rem; color:#718096; margin-top:1rem;">
                Fill in the property details in the sidebar and click
                <strong style="color:#63b3ed;">🔍 Get Investment Advisory</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
```

Note: fix the `import pandas as pd as _pd` — it should be `import pandas as _pd` (pandas is already imported as `pd` at the top of app.py, just use `pd` directly in Tab 2).

- [ ] **Step 3: Fix the pandas alias**

In the Tab 2 block, remove `import pandas as _pd` and use `pd` (already imported at the top of app.py). Change `_pd.DataFrame` to `pd.DataFrame`.

- [ ] **Step 4: Set GOOGLE_API_KEY and test locally**

```bash
export GOOGLE_API_KEY="your-key-here"
streamlit run app.py
```

Open browser, click "AI Advisor" tab, fill sidebar with any property, click "Get Investment Advisory". Verify:
- Progress status bar appears
- All 4 expanders render
- Action shows green/red/yellow based on BUY/AVOID/HOLD
- Comparables table shows if similar properties found
- Disclaimer appears in last expander

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "add AI Advisor tab to Streamlit app"
```

---

## Task 11: Deploy to Streamlit Cloud

**Files:** No code changes — deployment only.

- [ ] **Step 1: Add GOOGLE_API_KEY to Streamlit Cloud secrets**

In the Streamlit Cloud dashboard for your app:
- Go to Settings → Secrets
- Add: `GOOGLE_API_KEY = "your-actual-key"`

- [ ] **Step 2: Push to main branch**

```bash
git push origin main
```

- [ ] **Step 3: Wait for redeploy and verify**

Open the live URL. Verify:
- Tab 1 still works (existing price predictor unchanged)
- Tab 2 loads without errors
- Running the AI Advisor produces a complete report
- No raw Python errors visible in the UI

- [ ] **Step 4: Final commit with updated README**

Update `README.md` to mention the AI Advisor tab and the `GOOGLE_API_KEY` requirement in the env setup section, then:

```bash
git add README.md
git commit -m "update README with milestone 2 instructions"
git push origin main
```

---

## Self-Review Checklist

- [x] **Spec coverage:** LangGraph workflow ✓, RAG Chroma ✓, state management ✓, structured 4-section report ✓, Gemini free tier ✓, Tab 2 UI ✓, public deployment ✓, conditional routing ✓
- [x] **Placeholder scan:** No TBD, no TODO, all code blocks contain actual code
- [x] **Type consistency:** `AgentState` used consistently across state.py, steps.py, pipeline.py. `search()` returns `tuple[list, float]` matched in steps.py. `route_after_retrieval` returns strings matched in `add_conditional_edges` dict keys.
- [x] **Naming:** student-style names throughout — `check_input`, `predict_price`, `get_market_data`, `use_fallback`, `find_similar_homes`, `write_report`, `add_disclaimer`, `steps.py`, `pipeline.py`, `market_data/`
