import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Early Diabetes Risk Prediction System")
st.write("Non-clinical AI-based risk assessment tool")

st.sidebar.header("Enter Patient Details")

preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glu = st.sidebar.number_input("Glucose Level", 50, 300, 120)
bp = st.sidebar.number_input("Blood Pressure", 30, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 10, 100, 30)

input_data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

prob = model.predict_proba(input_scaled)[0][1]

if prob < 0.3:
    risk = "Low Risk"
elif prob < 0.6:
    risk = "Medium Risk"
else:
    risk = "High Risk"

st.subheader("Prediction Result")
st.write(f"🧪 **Risk Probability:** {prob:.2f}")
st.write(f"🚨 **Risk Level:** {risk}")
