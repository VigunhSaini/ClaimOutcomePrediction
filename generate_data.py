"""
generate_data.py
Simulates realistic historical insurance claims data with concentrated, high-signal subgroup rules.
"""

import numpy as np
import pandas as pd


def generate_claims_data(n_rows: int = 450, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    payers = ["Aetna", "Cigna", "UnitedHC", "BlueCross"]
    providers = ["Dr. Rao", "Dr. Shah", "Dr. Lee", "Dr. Patel"]
    cpt_codes = ["99213", "99214", "99215", "99203", "99204"]
    diagnosis_codes = ["M54.5", "J06.9", "E11.9", "I10", "Z00.00"]

    # Procedure baseline costs
    cpt_baseline_cost = {
        "99213": 120.0,
        "99214": 180.0,
        "99215": 250.0,
        "99203": 160.0,
        "99204": 230.0,
    }

    records = []
    statuses = ["Paid", "Partially_Paid", "Denied"]

    for claim_id in range(1, n_rows + 1):
        payer = np.random.choice(payers, p=[0.30, 0.25, 0.25, 0.20])
        provider = np.random.choice(providers, p=[0.25, 0.25, 0.25, 0.25])
        cpt = np.random.choice(cpt_codes, p=[0.35, 0.25, 0.15, 0.15, 0.10])
        diag = np.random.choice(diagnosis_codes, p=[0.25, 0.25, 0.20, 0.20, 0.10])

        # Patient age (pure demographic, no artificial correlation with claim outcome)
        patient_age = int(np.clip(np.random.normal(loc=46, scale=18), 5, 88))

        # Billed amount centered around procedure baseline with realistic variance
        base_cost = cpt_baseline_cost[cpt]
        if np.random.rand() < 0.15:
            # High billing outlier (e.g. 1.6x to 2.6x base)
            billed_amount = round(base_cost * np.random.uniform(1.6, 2.6), 2)
        else:
            billed_amount = round(max(40.0, np.random.normal(loc=base_cost, scale=22.0)), 2)

        cost_ratio = billed_amount / base_cost

        # Concentrated, distinct subgroup rules (creates strong, realistic decision boundaries):
        # 1. UnitedHC with high billed-to-CPT ratio (> 1.35) -> ~50% denial chance
        if payer == "UnitedHC" and cost_ratio > 1.35:
            p_paid, p_partial, p_denied = 0.25, 0.25, 0.50
        # 2. Aetna with high-intensity follow-up code 99215 -> ~45% partial-payment chance
        elif payer == "Aetna" and cpt == "99215":
            p_paid, p_partial, p_denied = 0.45, 0.45, 0.10
        # 3. Very high cost ratio (> 1.6) from any payer -> ~35% partial, ~25% denied
        elif cost_ratio > 1.6:
            p_paid, p_partial, p_denied = 0.40, 0.35, 0.25
        # 4. Moderate overbilling (> 1.25) -> partial payment bump
        elif cost_ratio > 1.25:
            p_paid, p_partial, p_denied = 0.58, 0.32, 0.10
        else:
            # Baseline for standard, well-behaved claims
            p_paid, p_partial, p_denied = 0.81, 0.14, 0.05

        probs = np.array([p_paid, p_partial, p_denied], dtype=float)
        probs /= probs.sum()

        status = np.random.choice(statuses, p=probs)

        records.append({
            "id": claim_id,
            "payer": payer,
            "provider": provider,
            "cpt_code": cpt,
            "diagnosis_code": diag,
            "billed_amount": billed_amount,
            "patient_age": patient_age,
            "status": status,
        })

    df = pd.DataFrame(records)
    return df


def verify_signal(df: pd.DataFrame):
    """Verifies that the generated features exhibit sharp, non-trivial outcome separation."""
    cpt_baseline = {"99213": 120.0, "99214": 180.0, "99215": 250.0, "99203": 160.0, "99204": 230.0}
    df = df.copy()
    df["cost_ratio"] = df["billed_amount"] / df["cpt_code"].map(cpt_baseline)
    df["ratio_bucket"] = pd.qcut(df["cost_ratio"], q=4, labels=["Q1 (Low)", "Q2 (Normal)", "Q3 (Moderate)", "Q4 (High)"])

    print("\n" + "=" * 55)
    print("SIGNAL VERIFICATION: Outcome Rates by Billed Ratio Quartile")
    print("=" * 55)
    ratio_summary = df.groupby("ratio_bucket", observed=False)["status"].value_counts(normalize=True).unstack(fill_value=0) * 100
    print(ratio_summary.round(1))

    print("\n" + "=" * 55)
    print("SIGNAL VERIFICATION: UnitedHC Denial Rate by High Ratio (> 1.35)")
    print("=" * 55)
    df["is_high_ratio"] = df["cost_ratio"] > 1.35
    uhc_summary = df[df["payer"] == "UnitedHC"].groupby("is_high_ratio")["status"].value_counts(normalize=True).unstack(fill_value=0) * 100
    print(uhc_summary.round(1))


if __name__ == "__main__":
    df_claims = generate_claims_data(n_rows=450, random_seed=42)
    output_path = "claims.csv"
    df_claims.to_csv(output_path, index=False)

    print(f"Generated {len(df_claims)} claims to {output_path}")
    print("\nClass distribution:")
    counts = df_claims["status"].value_counts()
    percentages = df_claims["status"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "percentage": percentages.round(1)})
    print(summary)

    verify_signal(df_claims)

    verify_signal(df_claims)

