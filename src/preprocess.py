"""
preprocess.py
-------------
Load, clean, encode, and split the churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Columns ───────────────────────────────────────────────────────────────────
CATEGORICAL = ["gender", "location", "contract",
               "payment_method", "internet_service"]
TARGET = "churn"
DROP_COLS = ["customer_id"]   # identifier – not a feature


def load_data(path: str = "data/customers.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing target (only if column exists)
    if TARGET in df.columns:
        df.dropna(subset=[TARGET], inplace=True)

    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Average revenue per month (sanity metric)
    df["avg_monthly_spend"] = (
        df["total_charges"] / df["tenure_months"].replace(0, 1)
    ).round(2)

    # High-value flag
    df["high_value"] = (df["monthly_charges"] > 80).astype(int)

    # Dissatisfied flag
    df["dissatisfied"] = (df["satisfaction_score"] <= 2).astype(int)

    # Service count (online_security + tech_support + streaming_tv + streaming_movies)
    df["service_count"] = (
        df["online_security"] + df["tech_support"] +
        df["streaming_tv"] + df["streaming_movies"]
    )

    return df


def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    encoders=None,
    scaler=None,
) -> tuple[pd.DataFrame, dict, StandardScaler]:
    """
    Encode categorical columns with LabelEncoder and scale numerics.

    Parameters
    ----------
    fit   : True  → fit + transform (training set)
            False → transform only  (test/new data)
    """
    df = df.copy()
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    if encoders is None:
        encoders = {}

    # Label-encode categoricals
    for col in CATEGORICAL:
        if col not in df.columns:
            continue
        le = encoders.get(col, LabelEncoder())
        if fit:
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[col] = le.transform(df[col].astype(str))

    # Scale numeric features (excluding target)
    feature_cols = [c for c in df.columns if c != TARGET]
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, encoders, scaler


def get_train_test(
    path: str = "data/customers.csv",
    test_size: float = 0.2,
    seed: int = 42,
):
    df_raw = load_data(path)
    df = clean_data(df_raw)
    df = feature_engineering(df)

    df_enc, encoders, scaler = encode_and_scale(df, fit=True)

    X = df_enc.drop(columns=[TARGET])
    y = df_enc[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, encoders, scaler
