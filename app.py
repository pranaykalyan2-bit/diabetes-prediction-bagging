import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------
# Load model and scaler
# ---------------------------------------------------

model = joblib.load("diabetes_bagging_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")


# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 35px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 15px;
}

.info-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    margin-top: 25px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

.disclaimer {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
    '<div class="title">🩺 Diabetes Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered diabetes risk assessment using Bagging Classification'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------
# Patient information
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=250,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.50
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


# ---------------------------------------------------
# Prediction button
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔍  Predict Diabetes",
    use_container_width=True
)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if predict_button:

    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)


    # Probability
    probability = model.predict_proba(input_scaled)


    # ------------------------------------------------
    # Result
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction[0] == 1:

        diabetes_probability = probability[0][1] * 100

        st.error(
            "⚠️ Higher Risk: Diabetes detected by the model."
        )

        st.metric(
            "Diabetes Probability",
            f"{diabetes_probability:.2f}%"
        )

    else:

        no_diabetes_probability = probability[0][0] * 100

        st.success(
            "✅ Lower Risk: No diabetes detected by the model."
        )

        st.metric(
            "No-Diabetes Probability",
            f"{no_diabetes_probability:.2f}%"
        )


# ---------------------------------------------------
# Model information
# ---------------------------------------------------

st.markdown(
    '<div class="info-card">'
    '<b>🤖 Model Information</b><br><br>'
    'Algorithm: Bagging Classifier<br>'
    'Base Estimator: Decision Tree<br>'
    'Number of Estimators: 100<br>'
    'OOB Score: 75.17%'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------
# Disclaimer
# ---------------------------------------------------

st.markdown(
    '<div class="disclaimer">'
    '⚕️ This prototype is intended for educational and demonstration '
    'purposes only and should not be used as a medical diagnosis.'
    '</div>',
    unsafe_allow_html=True
)