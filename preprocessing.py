"""
preprocessing.py
----------------
Standalone, reusable preprocessing pipeline for the House Price Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_pipeline(df: pd.DataFrame, target_col: str = 'price', test_size: float = 0.2):
    """
    Full preprocessing pipeline for house price data.

    Steps (in order):
        1. Drop non-informative columns (street, country)
        2. Parse 'date' column → year_sold, month_sold
        3. Engineer property_age from yr_built
        4. Handle missing values (median for numerical, mode for categorical)
        5. Remove invalid records (price/bedrooms/bathrooms == 0)
        6. Outlier treatment: quantile trim on price, IQR clip on other numerics
        7. Train-test split (80/20, random_state=42) ← BEFORE encoding/scaling
        8. OHE on city + statezip (fit on train, apply to both)
        9. StandardScaler on numerical columns (fit on train, apply to both)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from data.csv.
    target_col : str
        Name of the target column. Default: 'price'.
    test_size : float
        Fraction of data reserved for testing. Default: 0.2.

    Returns
    -------
    (X_train, X_test, y_train, y_test, scaler, feature_columns, log_transformed) : tuple
        - X_train, X_test   : pd.DataFrame – preprocessed feature sets
        - y_train, y_test   : pd.Series   – target values (log1p-transformed)
        - scaler            : fitted StandardScaler instance
        - feature_columns   : list[str]   – column names after all encoding/scaling
        - log_transformed   : bool        – always True; signals inverse np.expm1 needed
    """

    df = df.copy()

    # ------------------------------------------------------------------ #
    # STEP 1 – Drop non-informative columns
    # ------------------------------------------------------------------ #
    drop_cols = [c for c in ['street', 'country'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # ------------------------------------------------------------------ #
    # STEP 2 – Parse 'date' column
    # ------------------------------------------------------------------ #
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        df['year_sold']  = df['date'].dt.year
        df['month_sold'] = df['date'].dt.month
        df.drop(columns=['date'], inplace=True)

    # ------------------------------------------------------------------ #
    # STEP 3 – Engineer property_age
    # ------------------------------------------------------------------ #
    if 'yr_built' in df.columns:
        df['property_age'] = 2014 - df['yr_built']
        df.drop(columns=['yr_built'], inplace=True)

    # ------------------------------------------------------------------ #
    # STEP 4 – Handle missing values
    # ------------------------------------------------------------------ #
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ------------------------------------------------------------------ #
    # STEP 5 – Remove invalid records
    # ------------------------------------------------------------------ #
    if target_col in df.columns:
        df = df[df[target_col] != 0]
    if 'bedrooms' in df.columns:
        df = df[df['bedrooms'] != 0]
    if 'bathrooms' in df.columns:
        df = df[df['bathrooms'] != 0]

    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------ #
    # STEP 5b – Feature Engineering
    # ------------------------------------------------------------------ #
    df['total_sqft']      = df['sqft_living'] + df['sqft_basement']
    df['bed_bath_ratio']  = df['bedrooms'] / (df['bathrooms'] + 1)
    df['living_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)  # +1 avoids div/0
    if 'yr_renovated' in df.columns:
        df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
        df.drop(columns=['yr_renovated'], inplace=True)

    # ------------------------------------------------------------------ #
    # STEP 6 – Outlier treatment
    # ------------------------------------------------------------------ #
    # 6a. Quantile trim on price (1st–99th percentile) — removes rows
    if target_col in df.columns:
        lower_q = df[target_col].quantile(0.01)
        upper_q = df[target_col].quantile(0.99)
        df = df[(df[target_col] >= lower_q) & (df[target_col] <= upper_q)]
        df.reset_index(drop=True, inplace=True)

    # 6b. IQR clipping on all other numerical columns
    num_cols_current = df.select_dtypes(include=[np.number]).columns.tolist()
    other_num_cols = [c for c in num_cols_current if c != target_col]

    for col in other_num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # ------------------------------------------------------------------ #
    # STEP 6c – Log-transform target (after outlier trim, before split)
    # ------------------------------------------------------------------ #
    df[target_col] = np.log1p(df[target_col])

    # ------------------------------------------------------------------ #
    # STEP 7 – Train-test split (BEFORE encoding/scaling)
    # ------------------------------------------------------------------ #
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ------------------------------------------------------------------ #
    # STEP 8 – One-Hot Encoding on city & statezip
    # ------------------------------------------------------------------ #
    ohe_cols = [c for c in ['city', 'statezip'] if c in X_train.columns]

    if ohe_cols:
        X_train = pd.get_dummies(X_train, columns=ohe_cols, drop_first=True)
        X_test  = pd.get_dummies(X_test,  columns=ohe_cols, drop_first=True)

        # Align columns: test gets any columns missing relative to train, filled with 0
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # ------------------------------------------------------------------ #
    # STEP 9 – Standard scaling on numerical columns
    # ------------------------------------------------------------------ #
    sqft_cols = [
        'sqft_living', 'sqft_lot', 'sqft_above',
        'sqft_basement', 'sqft_living15', 'sqft_lot15'
    ]
    other_scale_cols = [
        'bedrooms', 'bathrooms', 'floors',
        'property_age', 'year_sold', 'month_sold'
    ]
    scale_candidates = sqft_cols + other_scale_cols
    scale_cols = [c for c in scale_candidates if c in X_train.columns]

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    # ------------------------------------------------------------------ #
    # STEP 10 – Collect feature column names
    # ------------------------------------------------------------------ #
    feature_columns = X_train.columns.tolist()

    return X_train, X_test, y_train, y_test, scaler, feature_columns, True


# --------------------------------------------------------------------------- #
# Quick smoke-test (run: python preprocessing.py)
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    result = preprocess_pipeline(df)
    print("X_train shape:", result[0].shape)
    print("X_test  shape:", result[1].shape)
    print("Features:", len(result[5]), "columns")
    print("Log-transformed:", result[6])
    print("Sample features:", result[5][:10])
