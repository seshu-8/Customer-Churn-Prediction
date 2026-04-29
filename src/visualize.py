"""
visualize.py
------------
All plots for EDA, model evaluation, and business insights.
Saved to outputs/ and images/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe everywhere)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

OUT  = "outputs"
IMG  = "images"
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

PALETTE = {"No Churn": "#2196F3", "Churn": "#F44336"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    path = f"{IMG}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── EDA plots ───────────────────────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame):
    counts = df["churn"].value_counts()
    labels = ["No Churn", "Churn"]
    colors = [PALETTE["No Churn"], PALETTE["Churn"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Churn Distribution (Pie)")

    sns.countplot(x="churn", data=df, palette=colors, ax=axes[1],
                  hue="churn", legend=False)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Churn Distribution (Count)")
    axes[1].set_xlabel("")

    fig.suptitle("Customer Churn Overview", fontsize=14, fontweight="bold")
    _save(fig, "01_churn_distribution")


def plot_churn_by_contract(df: pd.DataFrame):
    tbl = df.groupby("contract")["churn"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    tbl.plot(kind="bar", color="#E91E63", edgecolor="white", ax=ax)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(tbl):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    _save(fig, "02_churn_by_contract")


def plot_numeric_distributions(df: pd.DataFrame):
    num_cols = ["tenure_months", "monthly_charges", "total_charges",
                "satisfaction_score", "support_calls"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        for label, val in [(0, "No Churn"), (1, "Churn")]:
            subset = df[df["churn"] == label][col]
            axes[i].hist(subset, bins=30, alpha=0.6,
                         label=val, color=PALETTE[val], edgecolor="none")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].legend(fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle("Feature Distributions by Churn", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_numeric_distributions")


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    corr   = num_df.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    _save(fig, "04_correlation_heatmap")


# ─── Model evaluation plots ──────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    _save(fig, "05_confusion_matrix")


def plot_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#7B1FA2", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    _save(fig, "06_roc_curve")


def plot_feature_importance(model, feature_names: list[str]):
    if not hasattr(model, "feature_importances_"):
        print("  [skip] model has no feature_importances_")
        return

    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot(kind="barh", color="#00897B", edgecolor="none", ax=ax)
    ax.set_title("Feature Importances (Top 15)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    _save(fig, "07_feature_importance")


def plot_model_comparison(results_df: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df_plot = results_df[metrics].reset_index()
    df_melt = df_plot.melt(id_vars="index", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df_melt, x="Metric", y="Score",
                hue="index", ax=ax, palette="Set2")
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    _save(fig, "08_model_comparison")


def plot_churn_risk_segments(df_pred: pd.DataFrame):
    """Pie chart of low / medium / high churn risk."""
    bins   = [0, 0.33, 0.66, 1.0]
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    df_pred["risk_segment"] = pd.cut(df_pred["churn_probability"],
                                     bins=bins, labels=labels)
    counts = df_pred["risk_segment"].value_counts().reindex(labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = ["#66BB6A", "#FFA726", "#EF5350"]
    ax.pie(counts, labels=labels, autopct="%1.1f%%",
           colors=colors, startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("Customer Churn Risk Segments", fontsize=13, fontweight="bold")
    _save(fig, "09_churn_risk_segments")
