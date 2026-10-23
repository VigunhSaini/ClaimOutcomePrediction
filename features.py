"""
features.py
Custom scikit-learn transformer for feature engineering with zero data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClaimFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes domain-specific claim features and smoothed payer target statistics.
    Learns statistics strictly on training data (inside fit) to prevent data leakage.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy().reset_index(drop=True)

        if y is not None:
            y_series = pd.Series(y).reset_index(drop=True)

            # Global prior probabilities
            self.global_denial_rate_ = float((y_series == "Denied").mean())
            self.global_partial_rate_ = float((y_series == "Partially_Paid").mean())

            # Smoothed target encoding per payer (m-estimate)
            # rate = (count_class + m * global_prior) / (count_total + m)
            combined = pd.DataFrame({
                "payer": X_df["payer"],
                "is_denied": (y_series == "Denied").astype(int),
                "is_partial": (y_series == "Partially_Paid").astype(int),
            })

            grouped = combined.groupby("payer")
            payer_denials = grouped["is_denied"].sum()
            payer_partials = grouped["is_partial"].sum()
            payer_totals = grouped.size()

            m = self.smoothing
            self.payer_denial_rates_ = (
                (payer_denials + m * self.global_denial_rate_) / (payer_totals + m)
            ).to_dict()

            self.payer_partial_rates_ = (
                (payer_partials + m * self.global_partial_rate_) / (payer_totals + m)
            ).to_dict()
        else:
            # Fallback defaults if y is not provided
            self.global_denial_rate_ = 0.10
            self.global_partial_rate_ = 0.20
            self.payer_denial_rates_ = {}
            self.payer_partial_rates_ = {}

        # CPT code mean billed amount for relative cost ratios
        self.global_mean_billed_ = float(X_df["billed_amount"].mean())
        cpt_means = X_df.groupby("cpt_code")["billed_amount"].mean()
        self.cpt_mean_billed_ = cpt_means.to_dict()

        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy().reset_index(drop=True)

        # 1. Payer smoothed denial and partial rates with fallback to global prior
        X_out["payer_denial_rate"] = (
            X_out["payer"].map(self.payer_denial_rates_).fillna(self.global_denial_rate_).astype(float)
        )
        X_out["payer_partial_rate"] = (
            X_out["payer"].map(self.payer_partial_rates_).fillna(self.global_partial_rate_).astype(float)
        )

        # 2. Billed amount relative to average for that CPT code
        cpt_ref = X_out["cpt_code"].map(self.cpt_mean_billed_).fillna(self.global_mean_billed_).astype(float)
        X_out["billed_to_cpt_mean_ratio"] = (X_out["billed_amount"] / cpt_ref).astype(float)

        # 3. Log-transformed billed amount (compresses positive financial skew)
        X_out["log_billed_amount"] = np.log1p(np.maximum(0.0, X_out["billed_amount"])).astype(float)

        # 4. Patient age group (categorical)
        age = X_out["patient_age"]
        age_group = np.where(age <= 18, "child", np.where(age >= 65, "senior", "adult"))
        X_out["patient_age_group"] = age_group

        return X_out

