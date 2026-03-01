"""
run_training.py
---------------
Single orchestration script for the House Price Prediction project.

Run this ONCE to:
  1. Load and preprocess data.csv
  2. Train all three regression models
  3. Evaluate and compare models
  4. Auto-select the best model
  5. Generate assets/feature_importance.png
  6. Save all artifacts to models/

Usage:
  cd house_price_project/
  python run_training.py
"""

import pandas as pd

from preprocessing import preprocess_pipeline
from models import (
    train_all_models,
    evaluate_models,
    auto_select_best_model,
    model_selection_summary,
    plot_feature_importance,
    get_feature_insights,
    save_artifacts,
)

# ------------------------------------------------------------------ #
# STEP 1 — Load raw data
# ------------------------------------------------------------------ #
print("=" * 56)
print("  House Price Prediction — Training Pipeline")
print("=" * 56)
print("\n[1/7] Loading data.csv...")
df = pd.read_csv("data.csv")
print("      Rows: {:,}  |  Columns: {}".format(df.shape[0], df.shape[1]))

# ------------------------------------------------------------------ #
# STEP 2 — Preprocess
# ------------------------------------------------------------------ #
print("\n[2/7] Running preprocessing pipeline...")
X_train, X_test, y_train, y_test, scaler, feature_columns, log_transformed = preprocess_pipeline(df)
print("      X_train: {}  |  X_test: {}  |  Features: {}".format(
    X_train.shape, X_test.shape, len(feature_columns)
))

# ------------------------------------------------------------------ #
# STEP 3 — Train all models
# ------------------------------------------------------------------ #
print("\n[3/7] Training models...")
models = train_all_models(X_train, y_train)

# ------------------------------------------------------------------ #
# STEP 4 — Evaluate and print results table
# ------------------------------------------------------------------ #
print("\n[4/7] Evaluating models...")
results = evaluate_models(models, X_test, y_test, X_train, y_train)

# ------------------------------------------------------------------ #
# STEP 5 — Auto-select best model
# ------------------------------------------------------------------ #
print("\n[5/7] Selecting best model...")
best_name, best_model = auto_select_best_model(results)

print("\n--- Model Selection Summary ---")
print(model_selection_summary())

# ------------------------------------------------------------------ #
# STEP 6 — Feature importance chart
# ------------------------------------------------------------------ #
print("\n[6/7] Generating feature importance chart...")
plot_feature_importance(models, feature_columns)

insights = get_feature_insights()
print(insights)

# Persist text outputs so app.py can read them without in-memory cache
with open("assets/model_summary.txt", "w") as f:
    f.write(model_selection_summary())
print("  [saved] assets/model_summary.txt")

with open("assets/feature_insights.txt", "w") as f:
    f.write(insights)
print("  [saved] assets/feature_insights.txt")

# ------------------------------------------------------------------ #
# STEP 7 — Save all artifacts
# ------------------------------------------------------------------ #
print("\n[7/7] Saving artifacts...")
save_artifacts(
    best_model     = best_model,
    scaler         = scaler,
    feature_columns= feature_columns,
    all_models     = models,
)

# ------------------------------------------------------------------ #
# DONE
# ------------------------------------------------------------------ #
print("\n" + "=" * 56)
print("  All artifacts saved. Ready to run app.py")
print("=" * 56)

print("\nFiles saved:")
print("  models/best_regression_model.pkl")
print("  models/regression_scaler.pkl")
print("  models/column_reference.pkl")
print("  models/all_models.pkl")
print("  assets/feature_importance.png")
