import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('My Healthcare AI Predictor')
st.write('Enter the values below to get a prediction.')

# Example input — repeat st.number_input for each feature in your dataset
age = st.number_input('Age') 
sex = st.number_input('Sex') 
bmi = st.number_input('Body Mass Index(BMI)')
bp  = st.number_input('Blood Pressure')
s1  = st.number_input('Total Cholesterol')
s2  = st.number_input('Low Density Level Cholesterol(LDL)')
s3  = st.number_input('High Density Level Cholesterol(HDL)')
s4  = st.number_input('Cholesterol HDL Ratio')
s5  = st.number_input('Serum Triglycerides Level')
s6  = st.number_input('Blood Glucose Level')

if st.button('Predict'):
    input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)
    st.success(f'Prediction: {result[0]}')
