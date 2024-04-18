import streamlit as st
import pandas as pd
import pickle

# Load model
with open('pickle/pipeline_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Streamlit UI
st.title('Salary Prediction App')

# Input features
work_year = st.selectbox('Work Year', [2020, 2021, 2022, 2023])
experience_level = st.selectbox('Experience Level', ['SE', 'MI', 'EN', 'EX'])
job_title = st.selectbox('Job Title', ['Data Science & Machine Learning', 'Data Engineering & Infrastructure', 'Data Analysis & Analytics', 'Data Management', 'Artificial Intelligence (AI)', 'Business Intelligence (BI)'])
employee_residence = st.selectbox('Employee Residence', ['North America', 'Europe', 'Other'])
remote_ratio = st.selectbox('Remote Ratio', [100, 0, 50])
company_location = st.selectbox('Company Location', ['North America', 'Europe', 'Other'])
company_size = st.selectbox('Company Size', ['S', 'M', 'L'])

# Prepare input data
input_data = pd.DataFrame({
    'work_year': [work_year],
    'experience_level': [experience_level],
    'job_title': [job_title],
    'employee_residence': [employee_residence],
    'remote_ratio': [remote_ratio],
    'company_location': [company_location],
    'company_size': [company_size]
})

# Predict salary
if st.button('Predict Salary'):
    # Predict salary
    prediction = pipeline.predict(input_data)
    
    formatted_salary = '${:.2f}'.format(prediction[0])
    
    st.success(f'Predicted Salary: {formatted_salary}') 
