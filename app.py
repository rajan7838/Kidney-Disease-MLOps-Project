import streamlit as st
import joblib
import numpy as np


model = joblib.load("Models/best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")


st.set_page_config(page_title="Kidney Disease Predictor", layout="centered")
st.title("Chronic Kidney Disease Prediction")
st.write("Fill in the patient details below to check the likelihood of CKD.")


age = st.number_input("Age", min_value=0, max_value=120, value=50)
creatinine = st.number_input("Creatinine Level (mg/dL)", min_value=0.0, value=1.0, step=0.1)
bun = st.number_input("BUN (mg/dL)", min_value=0.0, value=15.0, step=0.1)


diabetes_option = st.selectbox("Diabetes", ["No", "Yes"])
hypertension_option = st.selectbox("Hypertension", ["No", "Yes"])

gfr = st.number_input("GFR (mL/min)", min_value=0.0, value=60.0, step=0.1)
urine_output = st.number_input("Urine Output (mL/24h)", min_value=0.0, value=1000.0, step=50.0)


diabetes = 1 if diabetes_option == "Yes" else 0
hypertension = 1 if hypertension_option == "Yes" else 0


if st.button("🔍 Predict CKD"):
    
    input_array = np.array([[age, creatinine, bun, diabetes, hypertension, gfr, urine_output]])
    
    
    input_scaled = scaler.transform(input_array)
    
    
    prediction = model.predict(input_scaled)[0]
    
    
    probability = model.predict_proba(input_scaled)[0][1]
    
    
    if prediction == 1:
        st.error(f"**CKD Detected** (Confidence: {probability:.1%})")
    else:
        st.success(f"**Healthy** (Confidence: {1-probability:.1%})")