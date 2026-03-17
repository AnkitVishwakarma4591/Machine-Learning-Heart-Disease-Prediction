import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ── Load model (cached so it only loads once per session) ──────────────────
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

loaded_model = load_model()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")
st.title("🫀 Heart Disease Prediction")
st.markdown("Fill in the details below and click **Predict** to check your 10-year CHD risk.")
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    sex = st.selectbox("Sex", ("Male", "Female"))
    bmi = st.number_input("BMI", min_value=1.0, max_value=70.0, value=25.0)
    current_smoker = st.selectbox("Current Smoker", ("No", "Yes"))
    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=1, max_value=250, value=75)
    glucose = st.number_input("Glucose Level", min_value=1.0, max_value=500.0, value=80.0)

with col2:
    sysBP = st.number_input("Systolic Blood Pressure", min_value=1.0, max_value=300.0, value=120.0)
    diaBP = st.number_input("Diastolic Blood Pressure", min_value=1.0, max_value=200.0, value=80.0)
    totChol = st.number_input("Total Cholesterol", min_value=1.0, max_value=700.0, value=200.0)
    BPMeds = st.selectbox("On Blood Pressure Medication", ("No", "Yes"))
    prevalentStroke = st.selectbox("Previous Stroke", ("No", "Yes"))
    prevalentHyp = st.selectbox("Hypertensive", ("No", "Yes"))
    diabetes = st.selectbox("Diabetes", ("No", "Yes"))

st.divider()

# ── Encode inputs ──────────────────────────────────────────────────────────
sex_enc             = 1 if sex == "Male" else 0
current_smoker_enc  = 1 if current_smoker == "Yes" else 0
BPMeds_enc          = 1 if BPMeds == "Yes" else 0
prevalentStroke_enc = 1 if prevalentStroke == "Yes" else 0
prevalentHyp_enc    = 1 if prevalentHyp == "Yes" else 0
diabetes_enc        = 1 if diabetes == "Yes" else 0

# Age normalization (same as training: divided by 75)
age_norm = age / 75

X_new = np.array([[age_norm, sex_enc, bmi, current_smoker_enc, heart_rate,
                   sysBP, diaBP, totChol, cigsPerDay,
                   BPMeds_enc, prevalentStroke_enc, prevalentHyp_enc,
                   diabetes_enc, glucose]])

# ── Predict ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):
    prediction = loaded_model.predict(X_new)[0]
    probability = loaded_model.predict_proba(X_new)[0][1] * 100

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Risk** — You are likely to develop CHD within 10 years.")
    else:
        st.success(f"✅ **Low Risk** — You are unlikely to develop CHD within 10 years.")

    st.metric(label="CHD Risk Probability", value=f"{probability:.1f}%")
    st.caption("⚠️ This is a predictive tool only. Please consult a medical professional for diagnosis.")