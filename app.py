import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('My Healthcare AI Predictor')
st.write('Enter the values below to get a prediction.')

# Example input — repeat st.number_input for each feature in your dataset
AGE = st.number_input('Age') 
SEX = st.number_input('Sex') 
BMI = st.number_input('Body Mass Index(BMI)')
BP  = st.number_input('Blood Pressure')
TC  = st.number_input('Total Cholesterol')
LDL = st.number_input('Low Density Level Cholesterol(LDL)')
HDL = st.number_input('High Density Level Cholesterol(HDL)')
TCH = st.number_input('Cholesterol HDL Ratio')
LTG = st.number_input('Serum Triglycerides Level')
GLU = st.number_input('Blood Glucose Level')

if st.button('Predict'):
    input_data = np.array([[age:Age, sex:Sex, bmi:BMI, bp:BP, s1:TC, s2:LDL, s3:HDL, s4:TCH, s5:LTG, s6:GLU]])
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)
    st.success(f'Prediction: {result[0]}')
