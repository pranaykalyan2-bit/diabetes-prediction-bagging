# Diabetes Prediction Using Bagging Technique

## 📌 Project Overview

This project is a Machine Learning application that predicts whether a person is likely to have diabetes based on medical information.

A **Bagging Classifier** with a **Decision Tree** as the base estimator is used for prediction. The trained model is integrated with a simple **Streamlit** web application where users can enter patient details and get a prediction.

## 🎯 Objective

The main objective of this project is to build a simple machine learning system that can predict diabetes based on important health-related features.

## 🧠 Machine Learning Model

The project uses:

- Bagging Classifier
- Decision Tree Classifier
- Feature Scaling
- Scikit-learn

Bagging combines multiple decision trees to improve the stability and performance of the prediction model.

## 📊 Input Features

The application takes the following patient information:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

```text
Diabetes prediction using bagging technique/
│
├── app.py
├── bagging_diabetes_prediction.ipynb
├── diabetes_bagging_model.pkl
├── diabetes_scaler.pkl
├── requirements.txt
└── README.md
