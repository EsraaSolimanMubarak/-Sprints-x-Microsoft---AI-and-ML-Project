import streamlit as st
import pandas as pd
import joblib

#Download the model and preprocessing tools
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

st.title(" Heart Disease Prediction App")

#Enter user data
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)", [1, 2, 3])

#Collect data in a DataFrame
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
    exang, oldpeak, slope, ca, thal
]], columns=[
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
])

# Same pipeline: Scaling → PCA → Predict
input_scaled = scaler.transform(input_data)
input_pca = pca.transform(input_scaled)
prediction = model.predict(input_pca)[0]

st.success(f" Predicted Heart Disease Class: {prediction}")