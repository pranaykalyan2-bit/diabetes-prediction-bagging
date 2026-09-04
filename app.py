import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("diabetes_bagging_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# Title
st.title("Diabetes Prediction")

st.write("Enter the patient details below:")

# Taking input from user
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

# Prediction button
if st.button("Predict"):

    # Create input data
    data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        pedigree,
        age
    ]]

    # Convert into DataFrame
    input_data = pd.DataFrame(data, columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    # Scale the data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Show result
    if prediction[0] == 1:
        st.error("The person may have diabetes.")
    else:
        st.success("The person may not have diabetes.")