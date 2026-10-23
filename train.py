"""
train.py
End-to-end training and evaluation script:
1. Stratified 80/20 train/test split
2. Simplified Random Forest pipeline (pruned noise features, regularized depth & leaves)
3. 360-row Out-Of-Fold (OOF) cross-validation evaluation (steadier than small test split)
4. Decision threshold tuning for minority Denied outcomes
5. Held-out test set evaluation (standard argmax vs. tuned threshold)
6. Defensible Permutation Feature Importance
7. Pipeline serialization to model.joblib
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import ClaimFeatureEngineer


def build_rf_pipeline() -> Pipeline:
    # Selected signal features (pruning provider, diagnosis_code, and patient_age which contain zero predictive signal)
    categorical_features = ["payer", "cpt_code"]
    numeric_features = [
        "billed_amount",
        "payer_denial_rate",
        "payer_partial_rate",
        "billed_to_cpt_mean_ratio",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("feature_engineer", ClaimFeatureEngineer(smoothing=10.0)),
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


def build_baseline_lr_pipeline() -> Pipeline:
    categorical_features = ["payer", "cpt_code"]
    numeric_features = [
        "billed_amount",
        "log_billed_amount",
        "payer_denial_rate",
        "payer_partial_rate",
        "billed_to_cpt_mean_ratio",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    return Pipeline(
        steps=[
            ("feature_engineer", ClaimFeatureEngineer(smoothing=10.0)),
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


def apply_decision_threshold(probs: np.ndarray, classes: np.ndarray, denial_threshold: float = 0.28) -> np.ndarray:
    """
    Applies custom operating threshold for minority Denied outcomes.
    If P(Denied) >= threshold, flags as Denied; otherwise picks argmax of remaining classes.
    """
    denied_idx = np.where(classes == "Denied")[0][0]
    paid_idx = np.where(classes == "Paid")[0][0]
    partial_idx = np.where(classes == "Partially_Paid")[0][0]

    preds = []
    for p in probs:
        if p[denied_idx] >= denial_threshold:
            preds.append("Denied")
        elif p[partial_idx] >= p[paid_idx]:
            preds.append("Partially_Paid")
        else:
            preds.append("Paid")
    return np.array(preds)


def main():
    print("Loading data from claims.csv...")
    df = pd.read_csv("claims.csv")

    X = df.drop(columns=["id", "status"])
    y = df["status"]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Total dataset: {len(df)} rows | Train: {len(X_train)} | Test: {len(X_test)}")
    print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
    print("Test class balance:\n", y_test.value_counts(normalize=True).round(3))

    rf_pipe = build_rf_pipeline()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Steadier Metrics: Out-Of-Fold (OOF) cross_val_predict on all 360 training rows
    print("\n" + "=" * 60)
    print("OUT-OF-FOLD (OOF) EVALUATION ACROSS ALL 360 TRAINING ROWS")
    print("=" * 60)
    oof_probs = cross_val_predict(rf_pipe, X_train, y_train, cv=skf, method="predict_proba")
    classes = np.unique(y_train)  # scikit-learn sorts classes alphabetically: ['Denied', 'Paid', 'Partially_Paid']

    oof_default_preds = classes[np.argmax(oof_probs, axis=1)]
    print("\n[OOF Default Argmax Report (Threshold ~0.33)]")
    print(classification_report(y_train, oof_default_preds, digits=3))

    # Threshold optimization on Out-of-Fold training data (never peek at test data)
    best_thresh = 0.33
    best_denied_f1 = f1_score(y_train == "Denied", oof_default_preds == "Denied")
    for th in np.arange(0.20, 0.40, 0.02):
        th_preds = apply_decision_threshold(oof_probs, classes, denial_threshold=th)
        f1_d = f1_score(y_train == "Denied", th_preds == "Denied")
        if f1_d > best_denied_f1:
            best_denied_f1 = f1_d
            best_thresh = round(float(th), 2)

    print(f"Optimal OOF Denial Threshold: {best_thresh} (improves Denied F1 from {f1_score(y_train == 'Denied', oof_default_preds == 'Denied'):.3f} to {best_denied_f1:.3f})")
    oof_tuned_preds = apply_decision_threshold(oof_probs, classes, denial_threshold=best_thresh)
    print(f"\n[OOF Tuned Report (Threshold = {best_thresh})]")
    print(classification_report(y_train, oof_tuned_preds, digits=3))

    # 2. Baseline Comparison
    lr_pipe = build_baseline_lr_pipeline()
    lr_scores = cross_val_score(lr_pipe, X_train, y_train, cv=skf, scoring="f1_macro")
    print(f"Baseline Logistic Regression Mean CV Macro F1: {lr_scores.mean():.3f} (+/- {lr_scores.std():.3f})")

    # 3. Fit primary model on complete training set and evaluate held-out test set
    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION (90 ROWS)")
    print("=" * 60)
    rf_pipe.fit(X_train, y_train)

    test_probs = rf_pipe.predict_proba(X_test)
    test_default_preds = rf_pipe.predict(X_test)
    test_tuned_preds = apply_decision_threshold(test_probs, rf_pipe.classes_, denial_threshold=best_thresh)

    labels = ["Paid", "Partially_Paid", "Denied"]

    print("\n--- Test Set: Default Argmax ---")
    print(classification_report(y_test, test_default_preds, labels=labels, digits=3))
    print("Confusion Matrix (Default):")
    cm_default = confusion_matrix(y_test, test_default_preds, labels=labels)
    print(pd.DataFrame(cm_default, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))

    print(f"\n--- Test Set: With Tuned Denial Threshold ({best_thresh}) ---")
    print(classification_report(y_test, test_tuned_preds, labels=labels, digits=3))
    print("Confusion Matrix (Tuned):")
    cm_tuned = confusion_matrix(y_test, test_tuned_preds, labels=labels)
    print(pd.DataFrame(cm_tuned, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))

    # 4. Permutation Feature Importance
    print("\n" + "=" * 60)
    print("PERMUTATION FEATURE IMPORTANCE (Defensible on Held-Out Test Set)")
    print("=" * 60)
    perm = permutation_importance(rf_pipe, X_test, y_test, n_repeats=15, random_state=42, scoring="f1_macro")
    perm_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance (Macro F1 Drop)": perm.importances_mean.round(4),
        "Std Dev": perm.importances_std.round(4),
    }).sort_values("Importance (Macro F1 Drop)", ascending=False)
    print(perm_df.to_string(index=False))

    # Save model pipeline
    model_filename = "model.joblib"
    joblib.dump(rf_pipe, model_filename)
    print(f"\nSaved simplified production pipeline to {model_filename}")


if __name__ == "__main__":
    main()
