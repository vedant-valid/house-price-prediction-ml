<div align="center">

![home](https://github.com/user-attachments/assets/1c924573-5cc8-4673-a686-be59fda38e69)

# Property Price Estimator
### ML-Based Valuation System — Washington State Housing Market

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://house-price-prediction-ml-p-9.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

*We started with a simple question: can a machine learn what makes one house worth twice as much as another?*

*Turns out, it can — and it comes down to a lot more than just the number of bedrooms.*

</div>

---

## Overview

This project is a complete machine learning system that estimates residential property sale prices using structured housing transaction data from the **greater Seattle metro area (2014 sales records)**.

Rather than stopping at a notebook, we built a fully modular pipeline — separate files for preprocessing, training, inference, and UI — so the whole system can be retrained, updated, or extended without touching the app.

| | |
|---|---|
| **Live Demo** | [house-price-prediction-ml-p-9.streamlit.app](https://house-price-prediction-ml-p-9.streamlit.app/) |
| **Data Source** | Residential sales records across 44 municipalities, 77 postal zones |
| **Records Used** | 4,458 valid transactions after cleaning |
| **Top Model** | Linear Regression — R² = **0.8028** after log-price optimization |

---

## Results at a Glance

> We ran five models and ranked them by test R² — no manual selection.

| Rank | Model | R² | Cross-Val R² | MAE | RMSE |
|:---:|---|:---:|:---:|---:|---:|
| 1 | **Linear Regression** | **0.8028** | **0.8160** | **$81,492** | **$138,943** |
| 2 | Ridge Regression | 0.7906 | 0.8083 | $84,840 | $143,394 |
| 3 | XGBoost | 0.7883 | 0.8056 | $85,287 | $142,682 |
| 4 | Random Forest | 0.7346 | 0.7572 | $96,655 | $159,300 |
| 5 | Decision Tree | 0.5193 | 0.5619 | $136,503 | $204,039 |

> The cross-validation column matters here — it confirms our R² of 0.8028 isn't a fluke of a lucky split. **The model generalizes.**

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

**The objective:** build a model that quantifies exactly how much each factor contributes to a final sale price.

---

## Dataset

Sales records covering residential transactions from a **3-month window in 2014–2015** spanning the broader Seattle region.

| Attribute | Detail |
|---|---|
| Total records | 4,600 |
| Valid after cleaning | 4,458 |
| Geographic spread | 44 cities, 77 zip codes |
| Features used | 18 raw → 135 after engineering |
| Target variable | Final sale price (USD) |

<details>
<summary><strong>Click to see core raw features</strong></summary>

```
living_area_sqft    lot_sqft         bedrooms
bathrooms           floors           waterfront
view_rating         condition        above_grade_sqft
basement_sqft       build_year       municipality
postal_zone
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
```

**The Log-Price Trick:**

```
Before log transform:  R² = 0.777
After  log transform:  R² = 0.8028  ← +2.6% from one line of code
```

Applying `np.log1p()` to the sale price target normalized the right-skewed distribution and made the regression loss function treat percentage errors consistently across cheap and expensive properties.

---

## Pipeline Architecture

```
data.csv  (raw input)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  preprocessing.py                                        │
│   ├── Drop non-informative fields                       │
│   ├── Parse date → year_sold, month_sold                │
│   ├── Engineer 4 new features                           │
│   ├── Remove invalid records (zero price, zero rooms)   │
│   ├── IQR clipping on feature columns                   │
│   ├── Quantile trim (1st–99th pct) on price only        │
│   ├── 80/20 train/test split  ← BEFORE encoding         │
│   ├── One-hot encode municipality + postal zone         │
│   ├── StandardScaler on continuous features             │
│   └── Log-transform price target                        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  models.py                                               │
│   ├── Train 5 models in parallel                        │
│   ├── Evaluate: R², 5-fold CV R², MAE, RMSE             │
│   ├── Auto-select winner by test R²                     │
│   └── Generate feature importance chart                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  inference.py                                            │
│   ├── predict_property(user_input: dict)                │
│   ├── Applies all transforms internally                 │
│   ├── Reverses log transform (np.expm1)                 │
│   └── Returns price, range, confidence, top drivers     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  app.py  (Streamlit UI)                                  │
│   ├── Sidebar input form                                │
│   ├── Calls predict_property()                          │
│   └── Displays prediction + chart + model rationale     │
└─────────────────────────────────────────────────────────┘
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/your-repo.git
cd house_price_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all 5 models (~2 mins)
python run_training.py

# 4. Launch the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## File Structure

```
house_price_project/
│
├── data.csv                         # Raw housing transaction data
├── House_Price_Prediction.ipynb     # Full analysis + EDA notebook
│
├── preprocessing.py                 # Modular preprocessing pipeline
├── models.py                        # Model training, eval, feature importance
├── inference.py                     # Single predict_property() entry point
├── run_training.py                  # One-command training orchestrator
├── app.py                           # Streamlit web application
├── requirements.txt                 # Pinned dependencies
│
├── models/
│   ├── best_regression_model.pkl
│   ├── regression_scaler.pkl
│   ├── column_reference.pkl
│   └── all_models.pkl
│
└── assets/
    ├── feature_importance.png
    ├── model_summary.txt
    └── feature_insights.txt
```

---

## Key Findings

> What actually drives Seattle home prices?

- **Interior living area** accounts for ~44% of total predictive weight — by far the strongest single signal
- **Geographic zone matters almost as much as size** — municipal and postal zone dummies contribute ~14% combined importance
- **Log-transforming the price target was the single biggest improvement** — more impactful than adding any new model
- Waterfront and high view ratings add measurable price premiums but aren't in the top 5 drivers overall
- **Linear beats nonlinear here** — once you're working in log-price space with 135 sparse OHE features, gradient boosting and tree ensembles actually underperform a well-regularized linear model

---

## Tech Stack

| Component | Library |
|---|---|
| Data wrangling | `pandas 2.2.2`, `numpy 1.26.4` |
| ML models | `scikit-learn 1.4.2`, `xgboost 2.0.3` |
| Visualization | `matplotlib 3.8.4`, `seaborn 0.13.2` |
| Model saving | `joblib 1.4.2` |
| Web app | `streamlit 1.33.0` |

---

## Team

| Member | Role |
|---|---|
| **Vedant Madne** [Lead] | Model training, evaluation, feature analysis |
| **Aryu Rao** | Data sourcing, preprocessing pipeline |
| **Vidhi Singhal** | Streamlit UI, cloud deployment |
| **Shitanshu Tiwari** | Notebook, documentation, report |

---

## Limitations Worth Knowing

| Limitation | Details |
|---|---|
| Temporal scope | Covers one 3-month window in 2014 — no awareness of post-pandemic price surges |
| Geographic scope | Trained exclusively on Seattle metro — not transferable to other markets |
| High-end accuracy | Avg error ~$81K — reliable for mid-market, less so above $1.5M |
| Input knowledge | Some fields (basement sqft, above-grade sqft) require detailed property knowledge |

---

## What's Next

- [x] Log transform on price
- [x] Ridge Regression
- [x] XGBoost
- [x] K-fold cross validation
- [ ] Incorporate more recent transaction data
- [ ] Add school district ratings and walkability scores as features
- [ ] REST API wrapper (FastAPI) for programmatic access
- [ ] Automated retraining when new data arrives

---

## Disclaimer

> Price estimates are generated by a statistical model trained on 2014 sales data. Outputs are for **educational and exploratory purposes only** and do not constitute professional real estate or financial advice.

---

<div align="center">

*Project 9 — Milestone 1 | AI/ML Course*

</div>
