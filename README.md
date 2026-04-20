<div align="center">

![home](https://github.com/user-attachments/assets/1c924573-5cc8-4673-a686-be59fda38e69)

# Property Price Estimator + AI Real Estate Advisor
### ML Valuation · Agentic Advisory · Washington State Housing Market

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://house-price-prediction-ml-p-9.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-00C49A?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

*We started with a simple question: can a machine learn what makes one house worth twice as much as another?*

*Milestone 1 answered that. Milestone 2 asked the harder question — can it tell you whether you should buy it?*

</div>

---

## What This Project Does

Two milestones. One system.

**Milestone 1** is a machine learning pipeline that predicts residential sale prices from structured property data — square footage, location, condition, age. It trains five models, auto-selects the best one, and serves predictions through a Streamlit UI.

**Milestone 2** wraps that predictor inside an agentic AI advisory system. It doesn't just return a number — it retrieves relevant market context, finds comparable sales, and generates a grounded BUY / HOLD / AVOID investment recommendation using a large language model. The whole thing runs as a multi-step LangGraph pipeline with explicit state manage.

| | |
|---|---|
| **Live Demo** | [house-price-prediction-ml-p-9.streamlit.app](https://house-price-prediction-ml-p-9.streamlit.app/) |
| **Data Source** | Residential sales records — 44 municipalities, 77 postal zones |
| **Records Used** | 4,458 valid transactions after cleaning |
| **Best ML Model** | Linear Regression — R² = **0.8028** |
| **Agentic Stack** | LangGraph · TF-IDF RAG · Qwen-2.5-72B via HuggingFace |

---

# Milestone 1 — ML Price Predictor

## Results at a Glance

> We ran five models and ranked them by test R² — no manual selection.

| Rank | Model | R² | Cross-Val R² | MAE | RMSE |
|:---:|---|:---:|:---:|---:|---:|
| 1 | **Linear Regression** | **0.8028** | **0.8160** | **$81,492** | **$138,943** |
| 2 | Ridge Regression | 0.7906 | 0.8083 | $84,840 | $143,394 |
| 3 | XGBoost | 0.7883 | 0.8056 | $85,287 | $142,682 |
| 4 | Random Forest | 0.7346 | 0.7572 | $96,655 | $159,300 |
| 5 | Decision Tree | 0.5193 | 0.5619 | $136,503 | $204,039 |

> The cross-validation column matters — it confirms R² = 0.8028 isn't a fluke of a lucky split. **The model generalizes.**

---

## Problem We Solved

Property valuation is genuinely hard. Two homes on the same street can differ by $200K based on factors that aren't always obvious — interior square footage vs lot size, renovation history, floor level, proximity to water. Manual appraisals are slow and subjective.

We approached this as a supervised regression problem across four signal categories:

```
Structural signals  →  interior area, room counts, floor layout
Geographic signals  →  municipality, postal zone
Condition signals   →  property rating, view quality, waterfront access
Temporal signals    →  construction era, sale season
```

---

## Dataset

| Attribute | Detail |
|---|---|
| Total records | 4,600 |
| Valid after cleaning | 4,458 |
| Geographic spread | 44 cities, 77 zip codes |
| Features used | 18 raw → 135 after engineering |
| Target variable | Final sale price (USD) |
| Price range | $0 – $26,590,000 |
| Median price | $460,943 |

<details>
<summary><strong>Click to see core raw features</strong></summary>

```
sqft_living    sqft_lot       sqft_above     sqft_basement
bedrooms       bathrooms      floors         waterfront
view           condition      yr_built       yr_renovated
city           statezip       date
```
</details>

---

## Feature Engineering

The raw dataset is straightforward. What differentiated our pipeline was the **engineering layer** on top of it.

**New features we created:**
```python
total_sqft       = sqft_living + sqft_basement    # full interior volume
bed_bath_ratio   = bedrooms / (bathrooms + 1)     # room balance signal
living_lot_ratio = sqft_living / sqft_lot         # density indicator
is_renovated     = (yr_renovated > 0).astype(int) # binary upgrade flag
property_age     = 2014 - yr_built                # age at time of sale
```

**The Log-Price Trick:**

```
Before log transform:  R² = 0.777
After  log transform:  R² = 0.8028  ← +2.6% from one line of code
```

Applying `np.log1p()` to the target normalized the right-skewed distribution and made the model treat percentage errors consistently across cheap and expensive properties.

---

## ML Pipeline Architecture

```
data/data.csv  (raw input)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  logic/preprocessing.py                                  │
│   ├── Drop non-informative fields                       │
│   ├── Parse date → year_sold, month_sold                │
│   ├── Engineer 5 new features                           │
│   ├── Remove invalid records (zero price, zero rooms)   │
│   ├── IQR clipping on feature columns                   │
│   ├── Quantile trim (1st–99th pct) on price only        │
│   ├── 80/20 train/test split  ← BEFORE encoding         │
│   ├── One-hot encode city + statezip                    │
│   ├── StandardScaler on continuous features             │
│   └── Log-transform price target                        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  logic/models.py                                         │
│   ├── Train 5 models in parallel                        │
│   ├── Evaluate: R², 5-fold CV R², MAE, RMSE             │
│   ├── Auto-select winner by test R²                     │
│   └── Generate feature importance chart                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  logic/inference.py                                      │
│   ├── predict_property(user_input: dict)                │
│   ├── Applies all transforms internally                 │
│   ├── Reverses log transform (np.expm1)                 │
│   └── Returns price, ±10% range, confidence, drivers    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  app.py — Tab 1: Price Predictor                         │
│   ├── Sidebar input form                                │
│   ├── Calls predict_property()                          │
│   └── Displays prediction + chart + model rationale     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Findings

- **Interior living area** accounts for ~44% of total predictive weight — by far the strongest single signal
- **Geographic zone matters almost as much as size** — city and ZIP dummies contribute ~14% combined
- **Log-transforming the target was the single biggest improvement** — more impactful than adding any new model
- Waterfront and high view ratings add measurable premiums but aren't in the top 5 drivers overall
- **Linear beats nonlinear here** — in log-price space with 135 sparse OHE features, gradient boosting underperforms a well-regularized linear model

---

---

# Milestone 2 — Agentic AI Real Estate Advisor

A price estimate answers *what is this property worth?* Milestone 2 answers the harder question: *should I buy it?*

The advisory system doesn't make a single model call. It reasons step by step — validating inputs, predicting price, retrieving market context, finding comparable sales, and generating a structured investment report grounded in real data. The entire flow is managed by a LangGraph state machine.

---

## Agentic Pipeline — 7 Steps

```
Natural language query (optional)
    │
    ▼  LLM parses text → structured property dict
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1 — check_input                                    │
│   └── Validates required fields. Missing → exits early  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2 — predict_price                                  │
│   └── Calls logic/inference.py → stores predicted_price │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3 — get_market_data  (RAG)                         │
│   ├── Builds query from property inputs                 │
│   ├── TF-IDF cosine similarity over 10 knowledge files  │
│   └── Top-4 chunks → stored in retrieved_docs           │
└───────────────────────┬─────────────────────────────────┘
                        │
                   ┌────▼────┐
                   │ Router  │  retrieval_score < 0.01?
                   └────┬────┘
           ┌────────────┴────────────┐
        fallback                 continue
    (hardcoded facts)               │
           └────────────┬───────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4 — find_similar_homes                             │
│   ├── Scans data/data.csv for ±20% sqft, same beds/city │
│   └── Top 5 by price proximity → comparables            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5 — write_report  (LLM)                            │
│   ├── Prompt: property + RAG context + comparables      │
│   ├── Qwen-2.5-72B via HuggingFace InferenceClient      │
│   ├── Returns JSON: summary, action, market_notes       │
│   └── Retry x3 on rate limit · graceful fallback        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 6 — add_disclaimer                                 │
│   └── Appends responsible AI disclaimer to report       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  app.py — Tab 2: AI Advisor                              │
│   ├── Valuation summary + price range                   │
│   ├── Comparable properties table                       │
│   ├── BUY / HOLD / AVOID recommendation                 │
│   ├── Market notes                                      │
│   └── Disclaimer                                        │
└─────────────────────────────────────────────────────────┘
```

---

## RAG — Retrieval-Augmented Generation

LLMs hallucinate. Especially about real estate statistics. RAG fixes this by grounding every claim in retrieved facts before the model writes a single word.

**Knowledge base:** 10 plain-text files generated from the actual dataset covering —

| File | Content |
|---|---|
| `zip_price_stats.txt` | Median & mean price per ZIP code |
| `city_price_stats.txt` | Median price across 44 cities |
| `feature_impact.txt` | Price by sqft band, bedroom count, condition |
| `price_tier_analysis.txt` | Q1 / Q2 / Q3 / P90 thresholds |
| `neighborhood_rankings.txt` | Top 10 most & least expensive cities |
| `market_seasonality.txt` | Price variation by month |
| `waterfront_premium.txt` | Waterfront vs standard pricing |
| `king_county_overview.txt` | General market context |
| `investment_principles.txt` | Core real estate investment rules |
| `regulations_disclaimer.txt` | Legal & regulatory context |

**Retrieval:** TF-IDF vectorizer with bigrams + cosine similarity. No model download. No vector database. Instant at startup.

---

## Hallucination Reduction

The advisory prompt is engineered to prevent the LLM from making things up:

```
Decision Rules — follow these strictly:
- AVOID: condition <= 2, OR priced 10%+ above comparables,
         OR yr_built < 1960 with poor condition
- HOLD:  condition == 3 AND price within 5% of comparables,
         OR uncertain market signals
- BUY:   condition >= 4 AND price at or below comparable median,
         OR strong appreciation signals in context

Rules:
- Do NOT default to BUY. Most properties are HOLD.
- Cite actual numbers from the data in your reasoning.
- Do not invent statistics not present in the context.
- Respond ONLY with valid JSON.
```

---

## Natural Language Query

The AI Advisor tab accepts free-text input:

> *"I want a 3 bedroom house in Bellevue, 1500 sqft, built in 2005, good condition"*

The LLM parses this into a structured property dict, shows what it extracted, and feeds the result straight into the advisory pipeline. No manual sliders required.

---

## AgentState — How Data Flows

Every node reads from and writes to a single shared `AgentState` TypedDict:

```python
class AgentState(TypedDict):
    property_input:   dict          # raw user inputs
    predicted_price:  float         # ML model output
    price_range:      dict          # {"low": ..., "high": ...}
    retrieved_docs:   list[str]     # top-4 RAG chunks
    retrieval_score:  float         # cosine similarity
    comparables:      list[dict]    # similar homes from dataset
    report:           dict          # LLM advisory output
    error:            Optional[str]
```

State starts empty and fills as each node runs. If any step fails, the error field catches it cleanly — no unhandled crashes in the UI.

---

## Milestone 2 Tech Stack

| Component | Library / Service |
|---|---|
| Workflow orchestration | `langgraph` |
| RAG retrieval | `scikit-learn` TF-IDF |
| LLM inference | `huggingface_hub` · Qwen-2.5-72B-Instruct |
| State management | Python `TypedDict` |
| Prompt grounding | RAG context + structured JSON output |

---

---

# Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/vedant-valid/house-price-prediction-ml.git
cd house-price-prediction-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models + build knowledge base (~2 mins)
python run_training.py

# 4. Launch the app
streamlit run app.py
```

App opens at `http://localhost:8501`

**For the AI Advisor tab**, set your HuggingFace API key:
```bash
export HF_API_KEY="your-key-here"
```
Or add it to `.streamlit/secrets.toml`:
```toml
HF_API_KEY = "your-key-here"
```

---

# File Structure

```
house_price_project/
│
├── data/
│   ├── data.csv                     # Raw housing transaction data
│   └── market_data/                 # 10 RAG knowledge files
│       ├── city_price_stats.txt
│       ├── zip_price_stats.txt
│       ├── feature_impact.txt
│       └── ... (7 more)
│
├── logic/                           # All core Python logic
│   ├── preprocessing.py             # Modular preprocessing pipeline
│   ├── models.py                    # Model training, eval, selection
│   ├── inference.py                 # predict_property() entry point
│   ├── rag_builder.py               # Generates market_data/ from dataset
│   ├── pipeline.py                  # LangGraph state graph
│   ├── steps.py                     # 7 agent node functions
│   ├── rag.py                       # TF-IDF retrieval module
│   ├── prompts.py                   # LLM prompt templates
│   └── state.py                     # AgentState TypedDict
│
├── notebooks/
│   └── House_Price_Prediction.ipynb # EDA + full analysis
│
├── models/                          # Saved model artifacts
│   ├── best_regression_model.pkl
│   ├── regression_scaler.pkl
│   └── column_reference.pkl
│
├── assets/
│   ├── feature_importance.png
│   ├── model_summary.txt
│   └── feature_insights.txt
│
├── tests/
│   ├── test_steps.py                # Unit tests for agent nodes
│   └── test_pipeline.py             # Integration tests
│
├── report/
│   └── project_report.tex           # LaTeX end-sem report
│
├── app.py                           # Streamlit web application
├── run_training.py                  # One-command training orchestrator
└── requirements.txt
```

---

# Team

| Member | Role |
|---|---|
| **Vedant Madne** [Lead] | Agentic pipeline, LangGraph, RAG, LLM integration, deployment |
| **Aryu Rao** | Data sourcing, preprocessing pipeline |
| **Vidhi Singhal** | Streamlit UI, cloud deployment |
| **Shitanshu Tiwari** | Notebook, documentation, report |

---

# Limitations Worth Knowing

| Limitation | Details |
|---|---|
| Temporal scope | Trained on 2014 data — no awareness of post-pandemic price surges |
| Geographic scope | Seattle metro only — not transferable to other markets |
| High-end accuracy | MAE ~$81K — reliable for mid-market, less precise above $1.5M |
| LLM dependency | Advisory tab requires a HuggingFace API key; predictions still work without it |
| Free-tier quota | HuggingFace free tier has rate limits — heavy usage may need a paid plan |

---

# What's Next

**Milestone 1**
- [x] Log transform on price
- [x] Ridge Regression + XGBoost
- [x] 5-fold cross-validation
- [x] Auto model selection

**Milestone 2**
- [x] LangGraph agentic pipeline
- [x] RAG with TF-IDF retrieval
- [x] LLM investment advisory
- [x] Natural language query parsing
- [x] Streamlit Cloud deployment
- [ ] Incorporate more recent transaction data
- [ ] Add school district ratings and walkability scores
- [ ] Conversational memory across advisory turns
- [ ] REST API wrapper (FastAPI) for programmatic access

---

## Disclaimer

> Price estimates are generated by a statistical model trained on 2014 sales data. Advisory reports are AI-generated and grounded in retrieved market data. All outputs are for **educational and exploratory purposes only** and do not constitute professional real estate or financial advice. Always consult a licensed real estate professional before making investment decisions.

---

<div align="center">

*Project 9 — Milestone 1 + Milestone 2 | AI/ML Course*

</div>
