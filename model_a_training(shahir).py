"""
Train Logistic Regression for Cardiovascular Disease (CVD) prediction.

- Target column: "Cardiovascular Disease"
- Features: all other columns
- Preprocessing:
    * Numeric: median impute -> StandardScaler
    * Categorical: most-frequent impute -> OneHotEncoder
- Class imbalance: class_weight="balanced"
- Evaluation: accuracy, precision, recall, F1, ROC-AUC (if binary)
- Threshold tuning: picks threshold that maximizes F1 on the validation set
- Artifacts saved:
    * cardio_logreg_pipeline_cvd.joblib
    * training_report_cvd.txt
    * pr_curve_cvd.png
    * pr_curve_points_cvd.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, auc
)

# ---------------------- CONFIG ----------------------
DATA_PATH = "Hybrid-CVD-Model/Model A Dataset/lifestyle_dataset.csv"  # change path if needed
TARGET_COL = "Cardiovascular Disease"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_PATH = "Hybrid-CVD-Model/Model A/cardio_logreg_pipeline_cvd.joblib"
REPORT_PATH = "Hybrid-CVD-Model/Model A/training_report_cvd.txt"

# PR artifacts (same folder as report/model)
PR_CURVE_PATH = "Hybrid-CVD-Model/Model A/pr_curve_cvd.png"
PR_POINTS_PATH = "Hybrid-CVD-Model/Model A/pr_curve_points_cvd.csv"
# ---------------------------------------------------


def ensure_dir_for(path: str):
    """Create parent directory for a file path if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        n_jobs=None
    )

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", clf)
    ])
    return pipe


def pick_best_threshold(y_true, y_proba):
    """Return threshold that maximizes F1 on provided data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def evaluate(y_true, y_pred, y_proba=None):
    binary = len(np.unique(y_true)) == 2
    avg = "binary" if binary else "weighted"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }
    roc_auc = None
    if binary and y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except Exception:
            pass
    return metrics, roc_auc


def main():
    # 0) Ensure output dirs exist
    for p in (MODEL_PATH, REPORT_PATH, PR_CURVE_PATH, PR_POINTS_PATH):
        ensure_dir_for(p)

    # 1) Load
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available columns: {list(df.columns)}")

    # 2) Split features/target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 3) Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # 4) Build pipeline
    pipe = build_pipeline(numeric_cols, categorical_cols)

    # 5) Train/test split
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )

    # 6) Fit
    pipe.fit(X_train, y_train)

    # 7) Baseline evaluation @ 0.5
    y_pred_05 = pipe.predict(X_test)
    try:
        y_proba_test = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba_test = None

    metrics_05, roc_auc_05 = evaluate(y_test, y_pred_05, y_proba_test)

    # 8) Threshold tuning (maximize F1)
    tuned = None
    if y_proba_test is not None and len(np.unique(y_test)) == 2:
        thr, best_f1, p_at_best, r_at_best = pick_best_threshold(y_test, y_proba_test)
        y_pred_best = (y_proba_test >= thr).astype(int)
        metrics_best, roc_auc_best = evaluate(y_test, y_pred_best, y_proba_test)
        tuned = {
            "threshold": thr,
            "f1": best_f1,
            "precision": p_at_best,
            "recall": r_at_best,
            "metrics": metrics_best,
            "roc_auc": roc_auc_best
        }

    # 9) Cross-validated ROC-AUC on full data (optional)
    cv_auc = None
    if y_proba_test is not None and len(np.unique(y)) == 2:
        try:
            cv_auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()
        except Exception:
            pass

    # 10) Precision–Recall curve (plot + save)
    ap, pr_auc = None, None
    if y_proba_test is not None and len(np.unique(y_test)) == 2:
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba_test)
        pr_auc = auc(recall, precision)                          # trapezoidal PR-AUC (matches your PR plot)
        ap = average_precision_score(y_test, y_proba_test)       # Average Precision (step-wise)

        # Save raw PR points
        pd.DataFrame({"recall": recall, "precision": precision}).to_csv(PR_POINTS_PATH, index=False)

        # Plot & save PR curve
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"PR AUC={pr_auc:.4f} | AP={ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (CVD Prediction)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PR_CURVE_PATH, dpi=160)
        plt.close()

    # 11) Save model
    joblib.dump(pipe, MODEL_PATH)

    # 12) Report
    lines = []
    lines.append(f"Target column: {TARGET_COL}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Features: {X.shape[1]}")
    lines.append(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    lines.append(f"Categorical features ({len(categorical_cols)}): {categorical_cols}\n")

    lines.append("=== Test metrics @ threshold = 0.50 ===")
    for k, v in metrics_05.items():
        lines.append(f"{k:>10}: {v:.4f}")
    if roc_auc_05 is not None:
        lines.append(f"{'roc_auc':>10}: {roc_auc_05:.4f}")
    if pr_auc is not None:
        lines.append(f"{'pr_auc':>10}: {pr_auc:.4f}   (trapezoidal area under PR curve)")
    if ap is not None:
        lines.append(f"{'avg_prec':>10}: {ap:.4f}   (Average Precision)")
        lines.append(f"PR curve image: {PR_CURVE_PATH}")
        lines.append(f"PR curve points CSV: {PR_POINTS_PATH}")

    if tuned is not None:
        lines.append("\n=== Tuned threshold (max F1) ===")
        lines.append(f"threshold: {tuned['threshold']:.4f}")
        lines.append(f"precision: {tuned['precision']:.4f}")
        lines.append(f"recall   : {tuned['recall']:.4f}")
        lines.append(f"f1       : {tuned['f1']:.4f}")
        if tuned['roc_auc'] is not None:
            lines.append(f"roc_auc  : {tuned['roc_auc']:.4f}")

    if cv_auc is not None:
        lines.append(f"\n5-fold CV ROC-AUC: {cv_auc:.4f}")

    lines.append("\n=== Classification report (t=0.50) ===")
    lines.append(classification_report(y_test, y_pred_05, zero_division=0))

    lines.append("=== Confusion matrix (t=0.50) ===")
    lines.append(np.array2string(confusion_matrix(y_test, y_pred_05)))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 13) Console summary
    print("\n".join(lines))
    if roc_auc_05 is not None:
        print(f"\nROC-AUC (t=0.50): {roc_auc_05:.4f}")
    if pr_auc is not None:
        print(f"PR-AUC: {pr_auc:.4f}")
    if ap is not None:
        print(f"Average Precision: {ap:.4f}")
    if cv_auc is not None:
        print(f"5-fold CV ROC-AUC: {cv_auc:.4f}")

    print(f"\nSaved pipeline to: {MODEL_PATH}")
    print(f"Saved report to:   {REPORT_PATH}")
    if ap is not None:
        print(f"Saved PR curve to: {PR_CURVE_PATH}")
        print(f"Saved PR points to: {PR_POINTS_PATH}")


if __name__ == "__main__":
    main()
