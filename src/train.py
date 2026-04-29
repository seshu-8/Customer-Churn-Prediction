"""
train.py
--------
Train multiple classifiers, compare them, save the best one.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.preprocess import get_train_test

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42,
                                                      n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,
                                                           learning_rate=0.05,
                                                           random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05,
                                             eval_metric="logloss",
                                             random_state=42, n_jobs=-1),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }


def train_all(path: str = "data/customers.csv") -> tuple[object, pd.DataFrame]:
    X_train, X_test, y_train, y_test, encoders, scaler = get_train_test(path)

    # Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    results = {}
    trained  = {}

    for name, model in get_models().items():
        print(f"  Training → {name} …")
        model.fit(X_train_res, y_train_res)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        print(f"    F1={metrics['F1']}  ROC-AUC={metrics['ROC-AUC']}")

    results_df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)

    # Best model by ROC-AUC
    best_name  = results_df.index[0]
    best_model = trained[best_name]
    print(f"\n  Best model: {best_name}  (ROC-AUC={results_df.loc[best_name,'ROC-AUC']})")

    # Save
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(encoders,   f"{MODELS_DIR}/encoders.pkl")
    joblib.dump(scaler,     f"{MODELS_DIR}/scaler.pkl")
    print(f"  Model saved → {MODELS_DIR}/best_model.pkl")

    # Full report for best model
    y_pred = best_model.predict(X_test)
    print(f"\n{'='*55}")
    print(f"Classification Report — {best_name}")
    print('='*55)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return best_model, results_df, (X_test, y_test)
