import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Stroke Predictor", layout="centered")

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

MODEL_COLUMNS = [
    "gender","age","hypertension","heart_disease","ever_married",
    "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"
]

# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict(row):
    df = pd.DataFrame([row])

    # numeric columns (MUST be float)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["avg_glucose_level"] = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # ⚠️ IMPORTANT: KEEP THESE AS INTEGERS (training dtype)
    df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce").astype("Int64")
    df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce").astype("Int64")

    # categorical — stay strings
    cat_cols = [
        "gender","ever_married","work_type",
        "Residence_type","smoking_status"
    ]
    for c in cat_cols:
        df[c] = df[c].astype(str)

    # reorder to match training
    df = df[MODEL_COLUMNS]

    # debug view
    st.write("📌 DATA TO MODEL")
    st.write(df)
    st.write(df.dtypes)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= thr)
    return prob, pred


# ---------------------------
# UI
# ---------------------------
st.title("🩺 Stroke Risk Prediction")

gender = st.selectbox("Gender", ["Male","Female","Other"])
age = st.number_input("Age", 1, 120, 45)

hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", ["0","1"])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", ["0","1"])

ever_married = 1 if st.radio("Ever Married?", ["Yes", "No"]) == "Yes" else 0

work_type = st.selectbox(
    "Work Type",
    ["Private","Self-employed","Govt_job","children","Never_worked"]
)

Residence_type = st.selectbox("Residence Type", ["Urban","Rural"])

avg_glucose_level = st.number_input("Avg Glucose Level", 40.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 24.0)

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
    "smoking_status": smoking_status,
}

if st.button("🔮 Predict"):
    prob, pred = predict(features)

    st.success(f"Probability of Stroke: {prob:.3f}")
    if pred == 1:
        st.error("⚠️ High Risk — consult a doctor.")
    else:
        st.info("✅ Low Risk — keep healthy habits.")
