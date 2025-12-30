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

# ---------- STYLE ----------
st.markdown("""
<style>
body { background: linear-gradient(135deg, #e0f2fe, #fdf2f8); }

.card {
  background: rgba(255,255,255,.88);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 12px 30px rgba(0,0,0,.08);
  backdrop-filter: blur(6px);
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    padding: 10px;
    background: linear-gradient(135deg,#4f46e5,#3b82f6);
    color:white;
    border:0;
}
.stButton>button:hover { transform: scale(1.01); }

.center { text-align:center; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="card">
<h2 class="center">🧠 Stroke Risk Predictor</h2>
<p class="center" style="color:gray;">
Beautiful, simple — for awareness only (not medical advice).
</p>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = joblib.load("stroke_model.pkl")
thr = json.load(open("threshold.json"))["threshold"]

# ---------- MODEL COLUMNS ----------
MODEL_COLUMNS = [
    "gender","age","hypertension","heart_disease","ever_married",
    "work_type","Residence_type","avg_glucose_level","bmi","smoking_status",
]

# ---------- PREDICT ----------
def predict(input_data):
    df = pd.DataFrame([input_data])

    numeric_features = ["age","avg_glucose_level","bmi"]
    categorical_features = [
        "gender","hypertension","heart_disease",
        "ever_married","work_type","Residence_type","smoking_status"
    ]

    df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors="coerce")
    df[categorical_features] = df[categorical_features].astype(str)

    df = df[MODEL_COLUMNS]

    # 🔍 DEBUG — DO NOT REMOVE
    st.write("📌 DATAFRAME GOING TO MODEL")
    st.write(df)
    st.write("📌 COLUMN TYPES")
    st.write(df.dtypes)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= thr)
    return prob, pred


# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["🧑 Personal", "🩺 Health", "🏡 Lifestyle"])


# ------- TAB 1 -------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Personal Info")

    col1, col2 = st.columns(2)
    age = col1.slider("Age", 1, 100, 45)
    gender = col2.selectbox("Gender", ["Male", "Female", "Other"])

    ever_married = st.radio("Ever Married?", ["Yes", "No"])

    st.markdown("</div>", unsafe_allow_html=True)


# ------- TAB 2 -------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Health Profile")

    # keep categorical 0/1 as STRINGS (model expects categorical)
    hypertension = "1" if st.toggle("Hypertension") else "0"
    heart_disease = "1" if st.toggle("Heart Disease") else "0"

    avg_glucose = st.slider("Average Glucose Level", 40.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 60.0, 24.0)

    st.markdown("</div>", unsafe_allow_html=True)


# ------- TAB 3 -------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Lifestyle")

    work_type = st.selectbox(
        "Work Type",
        ["Private","Self-employed","Govt_job","children","Never_worked"]
    )

    res_type = st.selectbox("Residence Type", ["Urban","Rural"])

    smoking = st.selectbox(
        "Smoking Status",
        ["never smoked","formerly smoked","smokes","Unknown"]
    )

    predict_btn = st.button("✨ Predict Stroke Risk")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- RESULT ----------
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

    with st.spinner("Analyzing…"):
        time.sleep(1)

    st.markdown(f"""
    <div class="card">
        <h3 class="center">Risk Level: {prob_percent}%</h3>
    </div>
    """, unsafe_allow_html=True)

    if pred == 1:
        st.error("⚠️ High Stroke Risk — consult a medical professional.")
    else:
        st.success("💚 Low Stroke Risk — keep healthy habits!")

    st.markdown("""
    ### 💡 Tips
    • Stay active  
    • Control blood pressure  
    • Monitor glucose  
    • Avoid smoking  
    """)
