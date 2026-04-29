"""
generate_data.py
----------------
Synthetic customer dataset generator.
Simulates a telecom-style company with realistic churn behaviour.
"""

import numpy as np
import pandas as pd


def generate_churn_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Demographics ─────────────────────────────────────────────────────────
    age = rng.integers(18, 70, n_customers)
    gender = rng.choice(["Male", "Female"], n_customers)
    location = rng.choice(["Urban", "Suburban", "Rural"], n_customers,
                          p=[0.5, 0.35, 0.15])

    # ── Contract & billing ───────────────────────────────────────────────────
    contract = rng.choice(["Month-to-month", "One year", "Two year"],
                          n_customers, p=[0.55, 0.25, 0.20])
    tenure_months = rng.integers(1, 73, n_customers)
    monthly_charges = rng.uniform(20, 120, n_customers).round(2)
    total_charges = (monthly_charges * tenure_months +
                     rng.normal(0, 50, n_customers)).clip(0).round(2)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_customers, p=[0.34, 0.23, 0.22, 0.21]
    )
    paperless_billing = rng.choice([0, 1], n_customers, p=[0.4, 0.6])

    # ── Services ──────────────────────────────────────────────────────────────
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n_customers,
                                  p=[0.34, 0.44, 0.22])
    online_security = rng.choice([0, 1], n_customers, p=[0.5, 0.5])
    tech_support = rng.choice([0, 1], n_customers, p=[0.5, 0.5])
    streaming_tv = rng.choice([0, 1], n_customers, p=[0.4, 0.6])
    streaming_movies = rng.choice([0, 1], n_customers, p=[0.4, 0.6])
    num_products = rng.integers(1, 7, n_customers)

    # ── Support interactions ──────────────────────────────────────────────────
    support_calls = rng.poisson(1.5, n_customers)
    complaints = rng.poisson(0.5, n_customers)
    satisfaction_score = rng.integers(1, 6, n_customers)   # 1‒5

    # ── Churn probability (business logic) ───────────────────────────────────
    churn_score = np.zeros(n_customers)

    # Month-to-month contracts churn more
    churn_score += np.where(contract == "Month-to-month", 0.35, 0)
    churn_score += np.where(contract == "One year", 0.10, 0)

    # Short tenure → higher churn
    churn_score += np.where(tenure_months < 12, 0.20, 0)
    churn_score += np.where(tenure_months < 6,  0.15, 0)

    # High monthly charges → higher churn
    churn_score += np.where(monthly_charges > 80, 0.15, 0)
    churn_score += np.where(monthly_charges > 100, 0.10, 0)

    # Fiber optic without support → frustration
    churn_score += np.where(
        (internet_service == "Fiber optic") & (tech_support == 0), 0.15, 0)

    # Complaints & low satisfaction
    churn_score += complaints * 0.08
    churn_score += np.where(satisfaction_score <= 2, 0.20, 0)
    churn_score += np.where(satisfaction_score == 3, 0.08, 0)

    # Support calls (frustration signal)
    churn_score += np.where(support_calls >= 3, 0.12, 0)

    # Electronic check → historically higher churn
    churn_score += np.where(payment_method == "Electronic check", 0.08, 0)

    # No online security → discomfort
    churn_score += np.where(online_security == 0, 0.05, 0)

    # More products → more sticky
    churn_score -= (num_products - 1) * 0.03

    # Clip to valid probability range and add noise
    churn_prob = np.clip(churn_score + rng.normal(0, 0.05, n_customers), 0.05, 0.95)
    churn = (rng.random(n_customers) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id":        [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)],
        "age":                age,
        "gender":             gender,
        "location":           location,
        "tenure_months":      tenure_months,
        "contract":           contract,
        "payment_method":     payment_method,
        "paperless_billing":  paperless_billing,
        "monthly_charges":    monthly_charges,
        "total_charges":      total_charges,
        "internet_service":   internet_service,
        "online_security":    online_security,
        "tech_support":       tech_support,
        "streaming_tv":       streaming_tv,
        "streaming_movies":   streaming_movies,
        "num_products":       num_products,
        "support_calls":      support_calls,
        "complaints":         complaints,
        "satisfaction_score": satisfaction_score,
        "churn":              churn,
    })
    return df


if __name__ == "__main__":
    df = generate_churn_data()
    df.to_csv("data/customers.csv", index=False)
    print(f"Dataset saved → data/customers.csv  |  shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
