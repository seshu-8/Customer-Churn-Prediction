"""
predict.py
----------
Load saved model and predict churn for new customers.
"""

import joblib
import numpy as np
import pandas as pd
from src.preprocess import clean_data, feature_engineering, encode_and_scale, TARGET

MODELS_DIR = "models"


def load_artifacts():
    model    = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    encoders = joblib.load(f"{MODELS_DIR}/encoders.pkl")
    scaler   = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    return model, encoders, scaler


def predict_new_customers(df_new: pd.DataFrame) -> pd.DataFrame:
    model, encoders, scaler = load_artifacts()

    df = clean_data(df_new.copy())
    df = feature_engineering(df)
    df_enc, _, _ = encode_and_scale(df, fit=False,
                                    encoders=encoders, scaler=scaler)

    X = df_enc.drop(columns=[TARGET], errors="ignore")

    proba       = model.predict_proba(X)[:, 1]
    predictions = (proba >= 0.5).astype(int)

    result = df_new[["customer_id"]].copy()
    result["churn_prediction"]  = predictions
    result["churn_probability"] = proba.round(4)
    result["risk_level"]        = pd.cut(
        proba, bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return result


def demo_prediction():
    """Predict on 5 hand-crafted sample customers."""
    samples = pd.DataFrame({
        "customer_id":        ["DEMO001", "DEMO002", "DEMO003", "DEMO004", "DEMO005"],
        "age":                [23,  55,  34,  45,  28],
        "gender":             ["Male", "Female", "Male", "Female", "Male"],
        "location":           ["Urban", "Suburban", "Rural", "Urban", "Urban"],
        "tenure_months":      [2,   60,  15,  36,  3],
        "contract":           ["Month-to-month", "Two year", "Month-to-month",
                               "One year", "Month-to-month"],
        "payment_method":     ["Electronic check", "Credit card", "Electronic check",
                               "Bank transfer", "Electronic check"],
        "paperless_billing":  [1, 0, 1, 0, 1],
        "monthly_charges":    [95.0, 45.0, 75.0, 60.0, 105.0],
        "total_charges":      [190.0, 2700.0, 1125.0, 2160.0, 315.0],
        "internet_service":   ["Fiber optic", "DSL", "Fiber optic", "DSL", "Fiber optic"],
        "online_security":    [0, 1, 0, 1, 0],
        "tech_support":       [0, 1, 0, 1, 0],
        "streaming_tv":       [1, 0, 1, 0, 1],
        "streaming_movies":   [1, 0, 1, 0, 1],
        "num_products":       [2, 5, 3, 4, 1],
        "support_calls":      [4, 1, 3, 0, 5],
        "complaints":         [2, 0, 1, 0, 3],
        "satisfaction_score": [2, 5, 3, 4, 1],
    })

    results = predict_new_customers(samples)
    print("\n" + "="*60)
    print("DEMO PREDICTIONS")
    print("="*60)
    print(results.to_string(index=False))
    return results
