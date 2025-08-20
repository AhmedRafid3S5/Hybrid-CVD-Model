"""
Train Logistic Regression for Cardiovascular Disease (CVD) prediction with Feature Engineering.

- Target column: "Cardiovascular Disease"
- Features: all other columns (+ engineered features)
- Feature Engineering (auto, only if columns exist):
    * BMI from Height & Weight (handles cm vs m)
    * Age bins (young / middle / older / elderly)
    * Interactions: Age*Smoking, BMI*PhysicalActivity, Age*BMI
    * Lifestyle risk score from smoking, alcohol, physical activity
- Preprocessing:
    * Numeric: median impute -> StandardScaler
    * Categorical: most-frequent impute -> OneHotEncoder
- Class imbalance: class_weight="balanced"
- Evaluation: accuracy, precision, recall, F1, ROC-AUC (if binary)
- PR metrics: trapezoidal PR-AUC + Average Precision (AP)
- Threshold tuning: picks threshold that maximizes F1
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

from typing import Optional, Tuple, Dict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
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
DATA_PATH   = "Hybrid-CVD-Model/Model A Dataset/lifestyle_dataset.csv"  # change path if needed
TARGET_COL  = "Cardiovascular Disease"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

MODEL_PATH  = "Hybrid-CVD-Model/Model A/cardio_logreg_pipeline_cvd.joblib"
REPORT_PATH = "Hybrid-CVD-Model/Model A/training_report_cvd.txt"
PR_CURVE_PATH   = "Hybrid-CVD-Model/Model A/pr_curve_cvd.png"
PR_POINTS_PATH  = "Hybrid-CVD-Model/Model A/pr_curve_points_cvd.csv"
# ---------------------------------------------------


def ensure_dir_for(path: str):
    """Create parent directory for a file path if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------- FEATURE ENGINEERING ----------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Pandas-in / Pandas-out transformer.
    Attempts to detect common columns and create:
      - BMI from Height & Weight (unit-aware)
      - Age bins (young/middle/older/elderly)
      - Interactions: age*smoking, BMI*activity, age*BMI
      - Lifestyle risk score from smoking/alcohol/activity
    If a needed input column is missing, the feature is simply skipped.
    """

    def __init__(self):
        # store resolved column names discovered during fit
        self.cols_: Dict[str, Optional[str]] = {}

    @staticmethod
    def _find_col(cols, *keys) -> Optional[str]:
        """Find first column name containing any of the given key substrings (case-insensitive)."""
        cols_lc = {c.lower(): c for c in cols}
        for key in keys:
            for lc, orig in cols_lc.items():
                if key in lc:
                    return orig
        return None

    def fit(self, X: pd.DataFrame, y=None):
        # Resolve likely column names
        cols = list(X.columns)

        self.cols_["height"]  = self._find_col(cols, "height", "stature")
        self.cols_["weight"]  = self._find_col(cols, "weight", "mass")
        self.cols_["age"]     = self._find_col(cols, "age")
        self.cols_["smoke"]   = self._find_col(cols, "smok")     # smoking/smoker/smoke status
        self.cols_["alcohol"] = self._find_col(cols, "alco", "alcohol", "drink")
        self.cols_["activity"]= self._find_col(cols, "activity", "exercise", "workout", "physical")

        # Infer height unit if present
        self.height_in_cm_ = None
        hcol = self.cols_.get("height")
        if hcol is not None and pd.api.types.is_numeric_dtype(X[hcol]):
            med_h = np.nanmedian(X[hcol].values.astype(float))
            # crude heuristic: if median > 100, likely cm
            self.height_in_cm_ = (med_h > 100)

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        hcol = self.cols_.get("height")
        wcol = self.cols_.get("weight")
        acol = self.cols_.get("age")
        scol = self.cols_.get("smoke")
        lcol = self.cols_.get("alcohol")
        pcol = self.cols_.get("activity")

        # --- BMI ---
        if hcol is not None and wcol is not None:
            height = X[hcol].astype(float)
            weight = X[wcol].astype(float)
            height_m = height / 100.0 if self.height_in_cm_ else height
            with np.errstate(divide="ignore", invalid="ignore"):
                bmi = weight / (height_m ** 2)
            X["FE_BMI"] = bmi.replace([np.inf, -np.inf], np.nan)

        # --- Age Bins ---
        if acol is not None:
            age = X[acol].astype(float)
            # bins: [0,30), [30,45), [45,60), 60+
            X["FE_AgeBin"] = pd.cut(
                age,
                bins=[-np.inf, 30, 45, 60, np.inf],
                labels=["young", "mid", "older", "elderly"]
            )

        # Map helpers for lifestyle scoring (very tolerant)
        def _to_numeric_series(series):
            try:
                return pd.to_numeric(series, errors="coerce")
            except Exception:
                return pd.Series(index=series.index, dtype=float)

        # approximate encoders for smoking/alcohol/activity
        def _encode_smoking(s: pd.Series):
            if s is None:
                return None
            s_l = s.astype(str).str.lower()
            out = pd.Series(np.nan, index=s.index, dtype=float)
            out[s_l.str.contains("never|non", regex=True, na=False)] = 0
            out[s_l.str.contains("former|quit", regex=True, na=False)] = 0.5
            out[s_l.str.contains("occasion|light|sometimes", regex=True, na=False)] = 0.5
            out[s_l.str.contains("current|yes|daily|regular", regex=True, na=False)] = 1
            # fill numeric if column is numeric category
            num = _to_numeric_series(s)
            out = out.fillna(num)
            return out

        def _encode_alcohol(s: pd.Series):
            if s is None:
                return None
            s_l = s.astype(str).str.lower()
            out = pd.Series(np.nan, index=s.index, dtype=float)
            out[s_l.str.contains("never|no", regex=True, na=False)] = 0
            out[s_l.str.contains("social|occasion", regex=True, na=False)] = 0.5
            out[s_l.str.contains("daily|heavy|yes|regular", regex=True, na=False)] = 1
            num = _to_numeric_series(s)
            out = out.fillna(num)
            return out

        def _encode_activity(s: pd.Series):
            if s is None:
                return None
            s_l = s.astype(str).str.lower()
            out = pd.Series(np.nan, index=s.index, dtype=float)
            out[s_l.str.contains("sedentary|low|none", regex=True, na=False)] = 0
            out[s_l.str.contains("moderate|medium", regex=True, na=False)] = 0.5
            out[s_l.str.contains("high|active|vigorous", regex=True, na=False)] = 1
            num = _to_numeric_series(s)
            out = out.fillna(num)
            return out

        # Encoded proxies (if columns exist)
        smoke_enc = _encode_smoking(X[scol]) if scol is not None else None
        alco_enc  = _encode_alcohol(X[lcol]) if lcol is not None else None
        act_enc   = _encode_activity(X[pcol]) if pcol is not None else None

        # --- Lifestyle risk score (higher is worse): smoke + alcohol - activity ---
        if smoke_enc is not None or alco_enc is not None or act_enc is not None:
            se = smoke_enc if smoke_enc is not None else 0
            ae = alco_enc  if alco_enc  is not None else 0
            pe = act_enc   if act_enc   is not None else 0
            X["FE_LifestyleRisk"] = pd.Series(se, index=X.index).fillna(0) + \
                                    pd.Series(ae, index=X.index).fillna(0) - \
                                    pd.Series(pe, index=X.index).fillna(0)

        # --- Interactions ---
        # age * smoking
        if acol is not None and smoke_enc is not None:
            X["FE_Age_x_Smoke"] = pd.to_numeric(X[acol], errors="coerce") * pd.Series(smoke_enc, index=X.index)

        # BMI * activity
        if "FE_BMI" in X.columns and act_enc is not None:
            X["FE_BMI_x_Activity"] = X["FE_BMI"] * pd.Series(act_enc, index=X.index)

        # age * BMI
        if acol is not None and "FE_BMI" in X.columns:
            X["FE_Age_x_BMI"] = pd.to_numeric(X[acol], errors="coerce") * X["FE_BMI"]

        return X
# ---------------------------------------------------


def build_pipeline_with_fe() -> Pipeline:
    """
    Pipeline:
      1) FeatureEngineer (pandas in/out)
      2) ColumnTransformer (type-based selectors)
      3) LogisticRegression
    """
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, selector(dtype_include=np.number)),
            ("cat", cat_tf, selector(dtype_exclude=np.number)),
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
        ("fe", FeatureEngineer()),   # <-- NEW
        ("preprocess", pre),
        ("model", clf),
    ])
    return pipe


def pick_best_threshold(y_true, y_proba) -> Tuple[float, float, float, float]:
    """Return threshold that maximizes F1 on provided data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
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

    # 3) Build pipeline with Feature Engineering
    pipe = build_pipeline_with_fe()

    # 4) Train/test split
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )

    # 5) Fit (FE + preprocess + LR happen automatically)
    pipe.fit(X_train, y_train)

    # 6) Baseline evaluation @ 0.5
    y_pred_05 = pipe.predict(X_test)
    try:
        y_proba_test = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba_test = None

    metrics_05, roc_auc_05 = evaluate(y_test, y_pred_05, y_proba_test)

    # 7) Threshold tuning (maximize F1)
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

    # 8) Cross-validated ROC-AUC on full data (optional)
    cv_auc = None
    if y_proba_test is not None and len(np.unique(y)) == 2:
        try:
            cv_auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()
        except Exception:
            pass

    # 9) Precision–Recall curve (plot + save)
    ap, pr_auc = None, None
    if y_proba_test is not None and len(np.unique(y_test)) == 2:
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba_test)
        pr_auc = auc(recall, precision)                          # trapezoidal PR-AUC
        ap = average_precision_score(y_test, y_proba_test)       # Average Precision

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

    # 10) Save model
    joblib.dump(pipe, MODEL_PATH)

    # 11) Report
    lines = []
    lines.append(f"Target column: {TARGET_COL}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Features before FE: {X.shape[1]}")
    lines.append("Feature Engineering: BMI, Age bins, Lifestyle risk, Interactions (applied where source columns exist)\n")

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

    # 12) Console summary
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
