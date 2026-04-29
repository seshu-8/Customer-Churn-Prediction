"""
main.py
=======
Customer Churn Prediction — full pipeline runner.

Usage
-----
  python main.py              # run everything
  python main.py --predict    # skip training, only demo predictions
"""

import argparse
import os
import sys
import time

import pandas as pd

# ── ensure repo root is on path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.generate_data import generate_churn_data
from src.preprocess    import load_data, clean_data, feature_engineering
from src.train         import train_all
from src.predict       import demo_prediction
from src.visualize     import (
    plot_churn_distribution,
    plot_churn_by_contract,
    plot_numeric_distributions,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_model_comparison,
    plot_churn_risk_segments,
)

DATA_PATH = "data/customers.csv"


def banner(text: str, char: str = "═"):
    width = 60
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def step(n: int, label: str):
    print(f"\n[STEP {n}] {label}")


def main(predict_only: bool = False):
    t0 = time.time()

    banner("Customer Churn Prediction Pipeline", "═")
    print("Python version:", sys.version.split()[0])

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Generate synthetic dataset
    # ──────────────────────────────────────────────────────────────────────────
    step(1, "Generating synthetic customer data …")
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        df_raw = generate_churn_data(n_customers=5000, seed=42)
        df_raw.to_csv(DATA_PATH, index=False)
        print(f"  Dataset created → {DATA_PATH}  |  shape: {df_raw.shape}")
    else:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"  Dataset loaded  → {DATA_PATH}  |  shape: {df_raw.shape}")

    churn_rate = df_raw["churn"].mean()
    print(f"  Churn rate: {churn_rate:.2%}  "
          f"({df_raw['churn'].sum()} churned / {len(df_raw)} total)")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — EDA Visualisations
    # ──────────────────────────────────────────────────────────────────────────
    step(2, "Exploratory Data Analysis …")
    df_clean = clean_data(df_raw)
    df_fe    = feature_engineering(df_clean)

    plot_churn_distribution(df_fe)
    plot_churn_by_contract(df_fe)
    plot_numeric_distributions(df_fe)
    plot_correlation_heatmap(df_fe)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Train & evaluate models
    # ──────────────────────────────────────────────────────────────────────────
    if not predict_only:
        step(3, "Training models …")
        best_model, results_df, (X_test, y_test) = train_all(DATA_PATH)

        step(4, "Model evaluation plots …")
        plot_confusion_matrix(best_model, X_test, y_test)
        plot_roc_curve(best_model, X_test, y_test)
        plot_feature_importance(best_model, list(X_test.columns))
        plot_model_comparison(results_df)

        # Save results table
        results_df.to_csv("outputs/model_comparison.csv")
        print(f"\n  Model comparison saved → outputs/model_comparison.csv")
        print(f"\n{results_df.to_string()}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Demo predictions
    # ──────────────────────────────────────────────────────────────────────────
    step(5, "Running demo predictions on new customers …")
    preds = demo_prediction()
    preds.to_csv("outputs/demo_predictions.csv", index=False)
    print(f"\n  Predictions saved → outputs/demo_predictions.csv")

    # Risk segment plot
    plot_churn_risk_segments(preds)

    # ──────────────────────────────────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"Pipeline complete  ·  {elapsed:.1f}s", "─")
    print("  Outputs → outputs/")
    print("  Images  → images/")
    print("  Model   → models/best_model.pkl\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Prediction")
    parser.add_argument("--predict", action="store_true",
                        help="Skip training; only run demo predictions")
    args = parser.parse_args()
    main(predict_only=args.predict)
