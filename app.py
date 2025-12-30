import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
)

# ---------- LOAD MODEL ----------
model = joblib.load("stroke_model.pkl")
thr = json.load(open("threshold.json"))["threshold"]

# ---------- PREDICT FUNCTION ----------
def predict(input_data):

    df = pd.DataFrame([input_data])

    df = df.reindex(columns=list(model.feature_names_in_))

    num_cols = ["age","avg_glucose_level","bmi"]

    cat_cols = [
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    # 🔥 CRITICAL FIX — ensure categoricals are STRINGS
    df[cat_cols] = df[cat_cols].astype(str)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= thr)
    return prob, pred

# ---------- UI ----------
tab1, tab2, tab3 = st.tabs(["🧑 Personal","🩺 Health","🏡 Lifestyle"])

with tab1:
    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male","Female"])
    ever_married = st.selectbox("Ever Married?", ["Yes","No"])

with tab2:
    hypertension = st.selectbox("Hypertension", [0,1])
    heart_disease = st.selectbox("Heart Disease", [0,1])
    avg_glucose = st.slider("Average Glucose Level", 40.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 60.0, 24.0)

with tab3:
    work_type = st.selectbox("Work Type",["Private","Self-employed","Govt_job","children","Never_worked"])
    res_type = st.selectbox("Residence Type",["Urban","Rural"])
    smoking = st.selectbox("Smoking Status",["never smoked","formerly smoked","smokes","Unknown"])
    predict_btn = st.button("✨ Predict Stroke Risk")

# ---------- PREDICT ----------
if predict_btn:

    features = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": res_type,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "smoking_status": smoking
    }

    prob, pred = predict(features)
    prob_percent = round(prob * 100, 1)

    st.info(f"Risk Level: {prob_percent}%")

    if pred == 1:
        st.error("⚠️ Higher stroke risk — consult a medical professional.")
    else:
        st.success("💚 Low stroke risk — keep healthy habits!")
