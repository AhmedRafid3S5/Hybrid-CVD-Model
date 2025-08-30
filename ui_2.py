import os
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CVD Stacking Predictor", page_icon="🫀", layout="centered")
st.title("🫀 Cardiovascular Disease – Stacked Predictor")
st.caption("Two LightGBM base models → Meta neural net → Final probability")

# -----------------------------
# File locations (put these next to app.py)
# -----------------------------
MODEL_A_PATH = "Saved Models A/lightgbm_model.txt"   # lifestyle features
MODEL_B_PATH = "Models B/lightgbm_model.txt"   # clinical features
META_MODEL_PATH = "simple_4_model.h5"       # Keras meta-model

# -----------------------------
# Expected master feature schema (names with underscores)
# Make sure these match your training dataset's column names.
# -----------------------------
MASTER_FEATURES = [
    "Age",
    "BMI",
    "Gender",                  # numeric encoding used during training
    "Alcohol_Intake",
    "Physical_Activity",
    "Smoking_Status",
    "Cholesterol_Level",
    "Systolic_Blood_Pressure",
    "Diastolic_Blood_Pressure",
    "Glucose_Level",
]

# CORRECTED: Model A expects the EXACT order it was trained on
# After dropping clinical features, the remaining lifestyle features should be in original dataset order
FEATURE_ORDER_A = ["Age", "Alcohol_Intake", "BMI", "Gender", "Physical_Activity", "Smoking_Status"]


# CORRECTED: Model B was trained on health/clinical features in original dataset order
# After dropping lifestyle features, the remaining clinical features should be:
FEATURE_ORDER_B = [
    "Age",                         # Column_0 [25:79]
    "BMI",                         # Column_1 [8:65.4]
    "Cholesterol_Level",           # Column_2 [1:3]  ← if you trained with Cholesterol first
    "Diastolic_Blood_Pressure",    # Column_3 [60:120]
    "Gender",                      # Column_4 [1:2]
    "Glucose_Level",               # Column_5 [1:3]  ← and Glucose second
    "Systolic_Blood_Pressure",     # Column_6 [80:200]
]

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    for p in [MODEL_A_PATH, MODEL_B_PATH, META_MODEL_PATH]:
        if not os.path.exists(p):
            st.error(f"Missing model file: {p}")
            st.stop()
    booster_A = lgb.Booster(model_file=MODEL_A_PATH)
    booster_B = lgb.Booster(model_file=MODEL_B_PATH)
    meta_model = load_model(META_MODEL_PATH)
    return booster_A, booster_B, meta_model

booster_A, booster_B, meta_model = load_models()

# -----------------------------
# Form UI
# -----------------------------
with st.form("inputs"):
    st.subheader("Enter features")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age (years)", min_value=0, max_value=120, value=45, step=1)
        BMI = st.number_input("BMI", min_value=0.0, max_value=80.0, value=27.5, step=0.1)
        Gender = st.selectbox("Gender (1 - female, 2 - male)", options=[1, 2], index=0,
                          help="Must match the encoding used in training (dataset used 1/2).")
        Alcohol_Intake = st.selectbox("Alcohol Intake (0/1)", options=[0, 1], index=0)
        Physical_Activity = st.selectbox("Physical Activity (0/1)", options=[0, 1], index=0)

    with col2:
        Smoking_Status = st.selectbox("Smoking Status (0/1)", options=[0, 1], index=0)
        Cholesterol_Level = st.selectbox("Cholesterol Level (1/2/3)", options=[1, 2, 3], index=0)
        Systolic_BP = st.number_input("Systolic_Blood_Pressure (mmHg)", min_value=0.0, value=120.0, step=1.0)
        Diastolic_BP = st.number_input("Diastolic_Blood_Pressure (mmHg)", min_value=0.0, value=80.0, step=1.0)
        Glucose_Level = st.selectbox("Glucose Level (1/2/3)", options=[1, 2, 3], index=0)

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Build one-row DataFrame
# -----------------------------
def build_master_row() -> pd.DataFrame:
    row = {
        "Age": Age,
        "BMI": BMI,
        "Gender": Gender,
        "Alcohol_Intake": Alcohol_Intake,
        "Physical_Activity": Physical_Activity,
        "Smoking_Status": Smoking_Status,
        "Cholesterol_Level": Cholesterol_Level,
        "Systolic_Blood_Pressure": Systolic_BP,
        "Diastolic_Blood_Pressure": Diastolic_BP,
        "Glucose_Level": Glucose_Level,
    }
    # Ensure all expected columns exist in this exact order
    return pd.DataFrame([[row[c] for c in MASTER_FEATURES]], columns=MASTER_FEATURES)

def make_subset_A(df_row: pd.DataFrame) -> pd.DataFrame:
    # Model A takes exactly these 6 columns in this order (per training)
    return df_row[FEATURE_ORDER_A]

def make_subset_B(df_row: pd.DataFrame) -> pd.DataFrame:
    # Model B takes exactly these 7 columns in this order (per training)
    return df_row[FEATURE_ORDER_B]

# -----------------------------
# Inference
# -----------------------------
if submitted:
    master = build_master_row()

    X_A = make_subset_A(master)
    X_B = make_subset_B(master)

    # LightGBM Booster.predict expects ndarray; returns raw_pred if not using predict_proba
    pA = float(booster_A.predict(X_A)[0])  # probability
    pB = float(booster_B.predict(X_B)[0])  # probability

    # Meta model expects shape (None, 2)
    meta_in = np.array([[pA, pB]], dtype=np.float32)
    p_final = float(meta_model.predict(meta_in, verbose=0)[0][0])

    st.markdown("---")
    st.subheader("Results")
    colL, colR = st.columns(2)
    with colL:
        st.metric("Model A (lifestyle) prob.", f"{pA * 100:.3f}%")
        st.metric("Model B (clinical) prob.", f"{pB * 100:.3f}%")
    with colR:
        st.metric("Final (meta) probability", f"{p_final * 100:.3f}%")
        st.write(
            "Interpretation: "
            + ("*High risk*" if p_final >= 0.5 else "*Low risk*")
            + " (threshold 0.50 – adjust per your use case)."
        )

    # Debug info to verify feature orders
    with st.expander("Debug: Feature Orders Used"):
        st.write("Model A features:", list(X_A.columns))
        st.write("Model B features:", list(X_B.columns))
        st.write("Model A values:", X_A.iloc[0].tolist())
        st.write("Model B values:", X_B.iloc[0].tolist())

    st.caption(
        "Note: Feature orders have been corrected to match the training dataset column order."
    )
