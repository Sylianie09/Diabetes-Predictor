import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('My Healthcare AI Predictor')
st.write('Enter the values below to get a prediction.')

# Example input — repeat st.number_input for each feature in your dataset
AGE = st.number_input('age') 
SEX = st.number_input('sex') 
BMI = st.number_input('bmi')
BP  = st.number_input('bp')
TC  = st.number_input('s1')
LDL = st.number_input('s2')
HDL = st.number_input('s3')
TCH = st.number_input('s4')
LTG = st.number_input('s5')
GLU = st.number_input('s6')

if st.button('Predict'):
    input_data = np.array([[GLU, BMI, BP, TC, LDL, HDL, TCH, LTG]])
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)
    st.success(f'Prediction: {result[0]}')
