"""
inference.py
------------
Single entry-point for all predictions in the House Price Prediction project.
Loads saved model artifacts once at import time; app.py only calls predict_property().

Artifacts expected (relative paths):
    models/best_regression_model.pkl
    models/regression_scaler.pkl
    models/column_reference.pkl
"""

import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# CONSTANTS
# ------------------------------------------------------------------ #
MODEL_PATH     = "models/best_regression_model.pkl"
SCALER_PATH    = "models/regression_scaler.pkl"
COLUMNS_PATH   = "models/column_reference.pkl"

# Numerical columns the scaler was fitted on (must match models.py / run_training.py)
SCALE_COLS = [
    "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
    "sqft_living15", "sqft_lot15",
    "bedrooms", "bathrooms", "floors",
    "property_age", "year_sold", "month_sold",
]

# ------------------------------------------------------------------ #
# LOAD ARTIFACTS (once, at import time)
# ------------------------------------------------------------------ #
def _load_artifact(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    _model           = _load_artifact(MODEL_PATH)
    _scaler          = _load_artifact(SCALER_PATH)
    _column_reference: list | None = _load_artifact(COLUMNS_PATH)   # ordered list of feature names
    _artifacts_ready = True
except FileNotFoundError as _e:
    _model = _scaler = _column_reference = None
    _artifacts_ready = False
    print(
        f"[inference.py] WARNING: Could not load artifact — {_e}\n"
        "  Run run_training.py first to generate model files."
    )


# ------------------------------------------------------------------ #
# HELPER — derive model metadata from loaded objects
# ------------------------------------------------------------------ #
def _model_name() -> str:
    """Return the class name of the loaded model."""
    if _model is None:
        return "Unknown"
    cls = type(_model).__name__
    name_map = {
        "LinearRegression":        "Linear Regression",
        "RandomForestRegressor":   "Random Forest",
        "DecisionTreeRegressor":   "Decision Tree",
    }
    return name_map.get(cls, cls)


def _confidence(r2: float = 0.733) -> str:
    """High if R² > 0.70, Medium otherwise."""
    return "High" if r2 > 0.70 else "Medium"


def _top_price_drivers() -> list:
    """
    Return top-3 feature names.
    Uses feature_importances_ for tree models, abs(coef_) for linear models.
    """
    if _model is None or _column_reference is None:
        return []
    try:
        if hasattr(_model, "feature_importances_"):
            importances = _model.feature_importances_
        elif hasattr(_model, "coef_"):
            importances = np.abs(_model.coef_)
        else:
            return []
        top_idx = np.argsort(importances)[::-1][:3]
        return [_column_reference[i] for i in top_idx]
    except Exception:
        return []


# ------------------------------------------------------------------ #
# MAIN PUBLIC FUNCTION
# ------------------------------------------------------------------ #
def predict_property(user_input: dict) -> dict:
    """
    Predict the sale price of a residential property.

    Parameters
    ----------
    user_input : dict
        {
          "sqft_living"  : float,
          "sqft_lot"     : float,
          "sqft_above"   : float,
          "sqft_basement": float,
          "bedrooms"     : int,
          "bathrooms"    : float,
          "floors"       : float,
          "waterfront"   : int,        # 0 or 1
          "view"         : int,        # 0–4
          "condition"    : int,        # 1–5
          "yr_built"     : int,
          "city"         : str,
          "statezip"     : str,
          "year_sold"    : int,        # default 2014
          "month_sold"   : int         # default 6
        }

    Returns
    -------
    dict with keys:
        predicted_price  : float
        price_range      : {"low": float, "high": float}
        top_price_drivers: list[str]   (top 3 feature names)
        model_used       : str
        confidence       : str         ("High" or "Medium")
    """
    if not _artifacts_ready:
        raise RuntimeError(
            "Model artifacts not loaded. Run run_training.py first."
        )

    # ---- a. Engineer property_age ---------------------------------- #
    yr_built     = int(user_input.get("yr_built", 1990))
    property_age = 2014 - yr_built

    # ---- b. Build base feature row --------------------------------- #
    city       = str(user_input.get("city", ""))
    statezip   = str(user_input.get("statezip", ""))
    year_sold  = int(user_input.get("year_sold",  2014))
    month_sold = int(user_input.get("month_sold", 6))

    base = {
        "sqft_living":   float(user_input.get("sqft_living",   0)),
        "sqft_lot":      float(user_input.get("sqft_lot",      0)),
        "sqft_above":    float(user_input.get("sqft_above",    0)),
        "sqft_basement": float(user_input.get("sqft_basement", 0)),
        "bedrooms":      float(user_input.get("bedrooms",      0)),
        "bathrooms":     float(user_input.get("bathrooms",     0)),
        "floors":        float(user_input.get("floors",        1)),
        "waterfront":    float(user_input.get("waterfront",    0)),
        "view":          float(user_input.get("view",          0)),
        "condition":     float(user_input.get("condition",     3)),
        "property_age":  float(property_age),
        "year_sold":     float(year_sold),
        "month_sold":    float(month_sold),
        # sqft_living15 / sqft_lot15 not collected from UI — default to sqft values
        "sqft_living15": float(user_input.get("sqft_living",   0)),
        "sqft_lot15":    float(user_input.get("sqft_lot",      0)),
    }

    row = pd.DataFrame([base])

    # ---- c. Apply OHE for city and statezip ------------------------ #
    # Build dummy column names from column_reference that start with city_ / statezip_
    city_prefix     = "city_"
    zip_prefix      = "statezip_"

    city_col     = city_prefix + city
    statezip_col = zip_prefix + statezip

    # Add zero-filled OHE columns for all cat columns in reference
    ohe_cols_in_ref = [
        c for c in _column_reference
        if c.startswith(city_prefix) or c.startswith(zip_prefix)
    ] if _column_reference is not None else []
    for col in ohe_cols_in_ref:
        row[col] = 0.0

    # Set the matching column to 1 only if it exists in training reference
    # (handles unseen city/statezip gracefully — stays 0, no KeyError)
    if _column_reference is not None and city_col in _column_reference:
        row[city_col] = 1.0

    if _column_reference is not None and statezip_col in _column_reference:
        row[statezip_col] = 1.0

    # ---- d. Scale numerical columns -------------------------------- #
    # sqft_living15 / sqft_lot15 are not collected from the UI;
    # use sqft_living and sqft_lot as proxies (nearest-neighbour avg ≈ subject property).
    row['sqft_living15'] = row['sqft_living']
    row['sqft_lot15']    = row['sqft_lot']

    # Only scale columns the fitted scaler actually knows about
    known_scale_cols = (
        list(_scaler.feature_names_in_)
        if _scaler is not None and hasattr(_scaler, 'feature_names_in_')
        else SCALE_COLS
    )
    scale_cols_present = [c for c in known_scale_cols if c in row.columns]
    if _scaler is not None:
        row[scale_cols_present] = _scaler.transform(row[scale_cols_present])

    # ---- e. Align to exact column_reference order ------------------ #
    # Add any missing columns as 0, drop any extra columns, reorder
    if _column_reference is not None:
        for col in _column_reference:
            if col not in row.columns:
                row[col] = 0.0

        row = row[_column_reference]

    # ---- f. Predict ------------------------------------------------ #
    # Model was trained on log1p(price) — invert with expm1 to get dollars
    if _model is None:
        raise RuntimeError(
            "Model not loaded. Run run_training.py first."
        )
    raw_prediction  = _model.predict(row)[0]
    predicted_price = float(np.expm1(raw_prediction))
    predicted_price = max(predicted_price, 0.0)   # safety floor

    low  = round(predicted_price * 0.90, 2)
    high = round(predicted_price * 1.10, 2)
    predicted_price = round(predicted_price, 2)

    return {
        "predicted_price": predicted_price,
        "price_range": {
            "low":  low,
            "high": high,
        },
        "top_price_drivers": _top_price_drivers(),
        "model_used":  _model_name(),
        "confidence":  _confidence(),
    }


# ------------------------------------------------------------------ #
# TEST BLOCK
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    sample = {
        "sqft_living":   1800,
        "sqft_lot":      5000,
        "sqft_above":    1800,
        "sqft_basement": 0,
        "bedrooms":      3,
        "bathrooms":     2.0,
        "floors":        1.0,
        "waterfront":    0,
        "view":          0,
        "condition":     3,
        "yr_built":      1990,
        "city":          "Seattle",
        "statezip":      "WA 98103",
        "year_sold":     2014,
        "month_sold":    6,
    }

    result = predict_property(sample)

    print("\n" + "=" * 48)
    print("  House Price Prediction — Inference Test")
    print("=" * 48)
    print("  Model used   : {}".format(result["model_used"]))
    print("  Confidence   : {}".format(result["confidence"]))
    print("  Predicted    : ${:,.0f}".format(result["predicted_price"]))
    print("  Range (90%)  : ${:,.0f}  –  ${:,.0f}".format(
        result["price_range"]["low"],
        result["price_range"]["high"],
    ))
    print("  Top drivers  : {}".format(", ".join(result["top_price_drivers"])))
    print("=" * 48)
