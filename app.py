import streamlit as st
import joblib
import json
import pandas as pd

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🩺")

st.title("Stroke Risk Prediction App 🧠")
st.write("""
This tool estimates stroke risk using a trained machine-learning model.  
**Educational use only — NOT medical advice.** 🩺
""")

# ---- LOAD MODEL + THRESHOLD ----
model = joblib.load("stroke_model.pkl")
thr = json.load(open("threshold.json"))["threshold"]

def predict(input_data):
    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= thr)
    return prob, pred


# ---- INPUT FORM ----
st.header("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

with col2:
    avg_glucose = st.number_input("Average Glucose Level", 40.0, 300.0)
    bmi = st.number_input("BMI", 10.0, 60.0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    work_type = st.selectbox("Work Type", [
        "Private", "Self-employed", "Govt_job", "children", "Never_worked"
    ])

res_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking = st.selectbox("Smoking Status", [
    "never smoked", "formerly smoked", "smokes", "Unknown"
])


# ---- PREDICT BUTTON ----
if st.button("Predict Stroke Risk"):
    features = {
        "age": age,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "gender": gender,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": res_type,
        "smoking_status": smoking
    }

    prob, pred = predict(features)

    st.subheader("Result 📊")
    st.info(f"Predicted Probability: **{prob:.3f}**")

    if pred == 1:
        st.error("High Risk — please consult a medical professional.")
    else:
        st.success("Low Risk — maintain regular check-ups.")
