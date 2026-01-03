import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load pickle files
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Disease Prediction (KNN)",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Disease Risk Prediction – KNN Model")
st.write("Enter realistic patient values to predict disease risk")

# -----------------------------
# INPUT FIELDS
# ⚠️ Change labels/order to EXACTLY match training columns
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=350, value=180)
glucose = st.number_input("Glucose Level", min_value=70, max_value=250, value=100)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    # Input must follow SAME order as training
    input_data = np.array([[age, bp, cholesterol, glucose]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict class
    prediction = model.predict(input_scaled)[0]

    # Predict probability (KNN supports this)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Disease Risk\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Disease Risk\n\nProbability: {probability:.2f}")
