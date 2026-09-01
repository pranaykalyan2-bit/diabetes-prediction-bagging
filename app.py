import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diabetes AI Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = joblib.load("diabetes_bagging_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1728 50%, #071827 100%);
    color: #ffffff;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main container */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hero */
.hero {
    padding: 45px 20px 35px 20px;
    text-align: center;
}

.hero-badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 30px;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(96, 165, 250, 0.35);
    color: #93c5fd;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #60a5fa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #94a3b8;
    font-size: 17px;
    margin-top: 14px;
}

/* Cards */
.card {
    background: rgba(15, 30, 50, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.18);
}

.card-title {
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 5px;
}

.card-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Metrics */
.metric-card {
    background: rgba(15, 30, 50, 0.85);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
    color: #60a5fa;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
}

/* Prediction */
.prediction-box {
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    background: rgba(15, 30, 50, 0.9);
    border: 1px solid rgba(96,165,250,0.25);
    margin-top: 20px;
}

.prediction-title {
    font-size: 30px;
    font-weight: 800;
}

.probability {
    font-size: 48px;
    font-weight: 800;
    margin: 10px 0;
    color: #60a5fa;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 52px;
    border: none;
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white;
    transition: 0.25s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(37,99,235,0.3);
}

/* Inputs */
div[data-baseweb="input"] > div {
    background-color: #0f1e32;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.15);
}

div[data-baseweb="select"] > div {
    background-color: #0f1e32;
    border-radius: 12px;
}

/* Section headings */
.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 35px;
    margin-bottom: 20px;
}

/* Info */
.info-box {
    padding: 18px;
    border-radius: 15px;
    background: rgba(30,41,59,0.6);
    border-left: 4px solid #3b82f6;
    color: #cbd5e1;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 50px;
    padding-top: 25px;
    border-top: 1px solid rgba(148,163,184,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown("""
<div class="hero">

<div class="hero-badge">
🤖 MACHINE LEARNING • BAGGING ENSEMBLE
</div>

<h1>Diabetes AI Analyzer</h1>

<p>
An interactive machine-learning system for estimating diabetes risk
using a Bagging Classifier.
</p>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------------
st.markdown('<div class="section-title">📊 Model Overview</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">77.60%</div>
        <div class="metric-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">75.79%</div>
        <div class="metric-label">5-Fold CV Score</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">75.17%</div>
        <div class="metric-label">OOB Score</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">100</div>
        <div class="metric-label">Decision Trees</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
st.markdown('<div class="section-title">👤 Patient Information</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="card">
<div class="card-title">Enter Patient Parameters</div>
<div class="card-subtitle">
Provide the patient's health information below to generate a model prediction.
</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "🤰 Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "🩸 Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "❤️ Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "📏 Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "💉 Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=80.0,
        value=32.0,
        step=0.1
    )

    pedigree = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01
    )

    age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=30
    )

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 ANALYZE DIABETES RISK"):

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0]

    diabetes_probability = probability[1] * 100

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    if prediction == 1:

        st.markdown(f"""
        <div class="prediction-box">

        <div class="prediction-title">
        ⚠️ Higher Diabetes Risk
        </div>

        <div class="probability">
        {diabetes_probability:.1f}%
        </div>

        <p style="color:#94a3b8;">
        Model-estimated probability of diabetes
        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="prediction-box">

        <div class="prediction-title">
        ✅ Lower Diabetes Risk
        </div>

        <div class="probability">
        {diabetes_probability:.1f}%
        </div>

        <p style="color:#94a3b8;">
        Model-estimated probability of diabetes
        </p>

        </div>
        """, unsafe_allow_html=True)

    # Probability bar
    st.markdown("### 🎯 Risk Probability")

    st.progress(int(diabetes_probability))

    st.caption(
        f"Estimated probability: {diabetes_probability:.2f}%"
    )

    # -----------------------------------------------------
    # INPUT SUMMARY
    # -----------------------------------------------------

    with st.expander("🔎 View Patient Data Used by the Model"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

# ---------------------------------------------------------
# MODEL EXPLANATION
# ---------------------------------------------------------
st.markdown('<div class="section-title">🧠 How the AI Works</div>',
            unsafe_allow_html=True)

with st.expander("🌳 What is Bagging?"):

    st.write("""
    Bagging (Bootstrap Aggregating) trains multiple decision trees
    using different bootstrap samples of the training data.

    Each tree makes a prediction and the final prediction is obtained
    by combining the predictions of the individual trees.

    In this project, the Bagging Classifier uses 100 decision trees.
    """)

with st.expander("📏 Why StandardScaler?"):

    st.write("""
    StandardScaler transforms the input features so that they are
    represented on a comparable scale.

    The same scaler used during model training is applied to new
    patient data before making a prediction.
    """)

with st.expander("🎯 What is OOB Score?"):

    st.write("""
    OOB (Out-of-Bag) score provides an internal estimate of model
    performance using samples that were not selected for training
    individual bootstrap models.
    """)

with st.expander("🔬 Why use Ensemble Learning?"):

    st.write("""
    Instead of depending on one decision tree, Bagging combines
    multiple trees.

    This can make the prediction more stable and reduce the
    effect of individual training samples on the final model.
    """)

# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------
st.markdown('<div class="section-title">⚡ Model Comparison</div>',
            unsafe_allow_html=True)

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Bagging Classifier",
        "Random Forest"
    ],
    "5-Fold CV Score": [
        71.23,
        75.79,
        76.18
    ]
})

st.bar_chart(
    comparison.set_index("Model")
)

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
st.markdown('<div class="section-title">📚 Dataset Information</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="info-box">

<b>Pima Indians Diabetes Dataset</b><br><br>

The model uses eight patient-related features:

Pregnancies • Glucose • Blood Pressure • Skin Thickness •
Insulin • BMI • Diabetes Pedigree Function • Age

The target variable is <b>Outcome</b>.

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

st.warning("""
⚠️ **Important:** This application is an educational machine-learning
prototype and is not a medical diagnostic tool. Predictions should not
be used as a substitute for professional medical advice.
""")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">

🩺 Diabetes AI Analyzer  
Built with Python • Scikit-learn • Streamlit • Bagging Ensemble Learning

</div>
""", unsafe_allow_html=True)