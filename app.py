import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")

# ---------------------------
# LOAD MODEL + THRESHOLD
# ---------------------------
@st.cache_resource
def load_model():
    model = joblib.load("stroke_model.pkl")
    with open("threshold.json") as f:
        thr = json.load(f)["threshold"]
    return model, thr

model, thr = load_model()

# These MUST match training order
MODEL_COLUMNS = [
    "gender","age","hypertension","heart_disease","ever_married",
    "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"
]


# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict(input_data):
    df = pd.DataFrame([input_data])

    # ---- FORCE DTYPE MATCHES (critical fix) ----
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["avg_glucose_level"] = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # convert binary flags into Yes/No (model was trained on strings)
    df["hypertension"] = df["hypertension"].astype(str).map(
        {"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"}
    )

    df["heart_disease"] = df["heart_disease"].astype(str).map(
        {"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"}
    )

    df["ever_married"] = df["ever_married"].astype(str).map(
        {"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"}
    )

    # Ensure all are strings (categorical safety)
    categorical_cols = [
        "gender","hypertension","heart_disease",
        "ever_married","work_type","Residence_type","smoking_status"
    ]
    for c in categorical_cols:
        df[c] = df[c].astype(str)

    # reorder strictly
    df = df[MODEL_COLUMNS]

    # Debug panel (helps if anything breaks)
    st.write("📌 DATA SENT TO MODEL")
    st.write(df)
    st.write(df.dtypes)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= thr)
    return prob, pred


# ---------------------------
# UI
# ---------------------------
st.title("🩺 Stroke Risk Prediction App")

st.write("Fill the details below to estimate risk of stroke 🚑")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=1, max_value=120, value=45)
hypertension = st.selectbox("Hypertension (BP)", ["0", "1"])
heart_disease = st.selectbox("Heart Disease", ["0", "1"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox(
    "Work Type",
    ["Private","Self-employed","Govt_job","children","Never_worked"]
)
Residence_type = st.selectbox("Residence Type", ["Urban","Rural"])
avg_glucose_level = st.number_input("Avg Glucose Level", min_value=40.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
smoking_status = st.selectbox(
    "Smoking Status",
    ["never smoked","formerly smoked","smokes","Unknown"]
)

features = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status
}

if st.button("🔮 Predict"):
    prob, pred = predict(features)

    st.success(f"Probability of Stroke: {prob:.3f}")
    if pred == 1:
        st.error("⚠️ High Risk — please consult a doctor.")
    else:
        st.info("✅ Low Risk — maintain healthy habits.")
