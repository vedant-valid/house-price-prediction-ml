# Milestone 2: Agentic AI Real Estate Advisory — Design Spec

**Date:** 2026-04-19  
**Project:** Intelligent Property Price Prediction & Agentic Real Estate Advisory (Project 9)  
**Status:** Approved

---

## Overview

Extend the existing Milestone 1 Streamlit app (ML price predictor) into an agentic AI real estate advisory assistant. A new "AI Advisor" tab is added to the deployed app. The agent uses LangGraph for workflow orchestration, Chroma for RAG-based market insight retrieval, and Google Gemini Flash (free tier) for natural language report generation.

---

## Architecture

```
app.py (Streamlit)
├── Tab 1: ML Price Predictor  (Milestone 1 — untouched)
└── Tab 2: AI Real Estate Advisor  (new)
         │
         ▼
   agent/graph.py  (LangGraph StateGraph)
         │
    AdvisoryState (TypedDict) flows through all nodes
         │
    Node 1: validate_input
    Node 2: predict_price         → calls inference.predict_property()
    Node 3: retrieve_market_info  → Chroma vector search
         │
    Conditional Edge:
      retrieval_score ≥ 0.4 → Node 4 (normal)
      retrieval_score < 0.4 → fallback_node → Node 4
         │
    Node 4: analyze_comparables   → top 5 similar from data.csv
    Node 5: generate_report       → Gemini Flash structured advisory
    Node 6: append_disclaimer     → fixed legal notice appended
         │
         ▼
   report: {summary, comps, action, disclaimer}
```

### New files
```
agent/
    __init__.py
    state.py      — AdvisoryState TypedDict
    nodes.py      — all 6 node functions + fallback
    graph.py      — LangGraph StateGraph wiring
    rag.py        — Chroma setup and retrieval helper
    prompts.py    — Gemini prompt templates
knowledge_base/   — 10 .txt knowledge files
chroma_db/        — pre-built vector store (committed to repo)
```

---

## LangGraph State

```python
class AdvisoryState(TypedDict):
    property_input:   dict         # raw user inputs
    predicted_price:  float        # from inference.py
    price_range:      dict         # {"low": float, "high": float}
    retrieved_docs:   list[str]    # chunks from Chroma
    retrieval_score:  float        # avg similarity of top-4 chunks
    comparables:      list[dict]   # top 5 similar properties from data.csv
    report:           dict         # {summary, comps, action, disclaimer}
    error:            str | None   # set by validate_input on bad input
```

## Node Responsibilities

| Node | Responsibility | Failure mode |
|---|---|---|
| `validate_input` | Check required fields present | Sets `error`; graph ends early |
| `predict_price` | Call `inference.predict_property()` | Falls back to dataset median |
| `retrieve_market_info` | Chroma top-4 chunk retrieval | Empty docs, score=0.0 |
| `fallback_node` | Inject dataset stats directly into state | Always succeeds |
| `analyze_comparables` | Filter data.csv for top 5 similar properties | Returns empty list |
| `generate_report` | Gemini Flash generates structured advisory | Returns generic fallback report |
| `append_disclaimer` | Append fixed legal/financial disclaimer | Always succeeds |

**Conditional edge** after `retrieve_market_info`:
- Score ≥ 0.4 → `analyze_comparables`
- Score < 0.4 → `fallback_node` → `analyze_comparables`

---

## RAG Pipeline

### Knowledge base (`knowledge_base/` — 10 files)

| File | Source |
|---|---|
| `king_county_overview.txt` | Hand-curated market overview |
| `investment_principles.txt` | Hand-curated buy/hold/avoid criteria |
| `waterfront_premium.txt` | Hand-curated waterfront stats |
| `regulations_disclaimer.txt` | Hand-curated WA state real estate rules |
| `zip_price_stats.txt` | Auto-generated from data.csv (median price per ZIP) |
| `city_price_stats.txt` | Auto-generated from data.csv (median price per city) |
| `feature_impact.txt` | Auto-generated from data.csv (sqft/bed/condition price impact) |
| `market_seasonality.txt` | Auto-generated from data.csv (month-of-sale patterns) |
| `price_tier_analysis.txt` | Auto-generated from data.csv (quartile breakdowns) |
| `neighborhood_rankings.txt` | Auto-generated from data.csv (top/bottom neighborhoods by price) |

### Build process (added to `run_training.py`)
1. Auto-generate 6 stats files from `data.csv`
2. Chunk all `.txt` files (500 tokens, 50 token overlap)
3. Embed with `sentence-transformers/all-MiniLM-L6-v2` (free, local)
4. Persist to `chroma_db/` → committed to repo

### Retrieval
- Query: `"{city} {statezip} {sqft_living}sqft {bedrooms}bed house price investment"`
- Returns top 4 chunks with similarity scores
- `retrieval_score` = average of top-4 similarity scores

---

## Report Output Structure

Four sections rendered in Streamlit as styled cards:

1. **Summary** — Property valuation, predicted price, market position relative to neighborhood median
2. **Comps** — Table of top 5 comparable properties (sqft, beds, price, delta%)
3. **Action** — BUY / HOLD / AVOID recommendation with 3–4 sentence Gemini-generated reasoning
4. **Disclaimer** — Fixed legal/financial notice (no LLM involvement)

---

## Streamlit UI (Tab 2)

- Property inputs reused from sidebar (same fields as Tab 1)
- "Get Investment Advisory" button triggers the LangGraph graph
- `st.status` progress indicator steps through nodes live
- Four collapsible `st.expander` sections for the report
- On `error` in state: red `st.error` banner with message, no raw traceback

---

## Deployment

- Same Streamlit Cloud app — redeploy with updated `app.py`
- `GOOGLE_API_KEY` stored as Streamlit Cloud secret (not committed)
- `chroma_db/` committed to repo (no embedding at runtime)
- New dependencies added to `requirements.txt`:
  - `langgraph`
  - `langchain-google-genai`
  - `langchain-community`
  - `chromadb`
  - `sentence-transformers`

---

## Rubric Coverage

| Criterion | How addressed |
|---|---|
| Agentic Reasoning & Decision Support | LangGraph multi-node graph with conditional routing |
| Correct RAG Integration | Chroma + sentence-transformers, score-gated retrieval |
| State Management | AdvisoryState TypedDict flows through all nodes |
| Utility of Advisory Insights | Structured 4-section report with comparables + recommendation |
| Responsible AI | Fixed disclaimer node, hallucination-reduction via RAG grounding |
| Deployment Quality | Same public Streamlit Cloud URL, no localhost |
