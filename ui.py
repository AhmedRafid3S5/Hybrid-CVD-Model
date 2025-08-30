import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

THRESHOLD = 0.50

# ---- Load model relative to this file ----
@st.cache_resource
def load_model():
    base = Path(__file__).resolve().parent
    model_path = base / "Model A" / "cardio_logreg_pipeline_cvd.joblib"
    if not model_path.exists():
        st.error(f"Model not found at:\n{model_path}")
        raise FileNotFoundError(model_path)
    return joblib.load(model_path)

pipe = load_model()
st.title("CVD Predictor (Streamlit)")

st.markdown(
    "Model expects these feature columns: "
    "`Age`, `Alcohol Intake`, `BMI`, `Gender`, `Physical Activity`, `Smoking Status`."
)

# ---------- Single prediction ----------
st.header("Single Prediction")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, value=50)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    gender = st.selectbox("Gender (1 - female, 2 - male)", options=[1, 2], index=0,
                          help="Must match the encoding used in training (dataset used 1/2).")
with col2:
    alcohol = st.selectbox("Alcohol Intake (0/1)", options=[0, 1], index=0)
    activity = st.selectbox("Physical Activity (0/1)", options=[0, 1], index=0)
    smoking = st.selectbox("Smoking Status (0/1)", options=[0, 1], index=0)

if st.button("Predict"):
    # Build EXACT columns the pipeline expects (no target)
    row = {
        "Age": age,
        "Alcohol Intake": alcohol,
        "BMI": bmi,
        "Gender": gender,
        "Physical Activity": activity,
        "Smoking Status": smoking,
    }
    X = pd.DataFrame([row])
    proba = float(pipe.predict_proba(X)[:, 1][0])
    pred = int(proba >= THRESHOLD)
    st.success(f"Probability of CVD: **{proba:.4f}**  |  Prediction (t={THRESHOLD:.2f}): **{pred}**")

# ---------- Batch prediction (CSV) ----------
st.header("Batch Prediction (CSV)")
st.caption(
    "CSV must contain these columns (exact names): "
    "`Age`, `Alcohol Intake`, `BMI`, `Gender`, `Physical Activity`, `Smoking Status`."
    " Do **not** include the target column."
)

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)

    required = {"Age", "Alcohol Intake", "BMI", "Gender", "Physical Activity", "Smoking Status"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
    else:
        probas = pipe.predict_proba(df[sorted(list(required))])[:, 1]
        preds = (probas >= THRESHOLD).astype(int)
        out = df.copy()
        out["CVD_Probability"] = probas
        out["CVD_Prediction"] = preds
        st.dataframe(out.head(20))
        st.download_button("Download predictions.csv", out.to_csv(index=False), "predictions.csv")
