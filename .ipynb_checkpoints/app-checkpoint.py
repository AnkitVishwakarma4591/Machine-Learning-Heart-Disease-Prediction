import pandas as pd
import numpy as np
import joblib
import streamlit as st

heart_disease_model_pkl = "heart_disease_model.pkl"

loaded_model = joblib.load(heart_disease_model_pkl)

st.header("Heart Disease Prediction Model")

age = st.number_input("Enter the Age ")

sex = st.selectbox("Enter the Sex ", ("male", "female"))
sex_dict = {"male": 1, "female": 0}
sex = sex_dict[sex]

bmi = st.number_input("Enter the BMI ")

current_smoker = st.selectbox("Do you currently Smoke ", ("no", "yes"))
current_smoker_dict = {"no": 0, "yes": 1}
current_smoker = current_smoker_dict[current_smoker]

heart_rate = st.number_input("Enter the Heart Rate in Beats per minute (BPM)")

sysBP = st.number_input("Enter the Systolic blood pressure")

diaBP = st.number_input("Enter the Diastolic blood pressure")

totChol = st.number_input("Enter your total cholesterol level ")

cigsPerDay = st.number_input("Enter the Number of cigarettes smoked per day ")

glucose = st.number_input("Enter your total glucose level ")

BPMeds = st.selectbox("You are on blood pressure medication", ("Yes", "No"))
BPMeds_dict = {"Yes": 1, "No": 0}
BPMeds = BPMeds_dict[BPMeds]

prevalentStroke = st.selectbox("Have you had a previous stroke", ("Yes", "No"))
prevalentStroke_dict = {"Yes": 1, "No": 0}
prevalentStroke = prevalentStroke_dict[prevalentStroke]

prevalentHyp = st.selectbox("You are hypertensive", ("Yes", "No"))
prevalentHyp_dict = {"Yes": 1, "No": 0}
prevalentHyp = prevalentHyp_dict[prevalentHyp]

diabetes = st.selectbox("Do you have diabetes", ("Yes", "No"))
diabetes_dict = {"Yes": 1, "No": 0}
diabetes = diabetes_dict[diabetes]

# FIX: Age normalization moved here, inside array construction
X_new = np.array([[age / 75, sex, bmi, current_smoker, heart_rate, sysBP, diaBP,
                   totChol, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, glucose]])

button = st.button("Submit")

# FIX: All prediction logic is now inside the button block only
if button:
    predicted_value = loaded_model.predict(X_new)
    decode_dict = {0: "No", 1: "Yes"}
    predicted_name = decode_dict[predicted_value[0]]

    if predicted_name == "Yes":
        st.warning(f"⚠️ Yes, you are at risk of developing CHD.")
    else:
        st.success(f"✅ No, you are not at risk of developing CHD.")