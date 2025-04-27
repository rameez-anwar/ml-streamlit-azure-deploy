import streamlit as st
import pandas as pd
import joblib

st.header("DIABETES PREDICTION")
model = joblib.load('model.pkl')

# Input Features
st.subheader("Input Features")
Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0, format="%.1f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
Age = st.number_input("Age", min_value=0, step=1)

 # Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age]
        })
    prediction = model.predict(input_data)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.subheader("Prediction Result")
    st.write(f"Prediction: {result}")