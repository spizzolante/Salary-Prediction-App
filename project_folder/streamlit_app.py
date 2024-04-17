from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import pickle

# Load model and preprocessing objects
with open('model.pkl', 'rb') as file:
    scaler, ct, pipeline = pickle.load(file)

def predict_salary(input_data):
    # Make predictions
    predictions = pipeline.predict(input_data)
    return predictions

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
    'company_size': [company_size],
    'salary_in_usd': [0]  # Placeholder value
})

# Predict salary
if st.button('Predict Salary'):
    # Transform input data
    input_data_transformed = ct.transform(input_data)

    original_categorical_columns = ['work_year', 'experience_level', 'job_title', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']
    original_categories = {
    'work_year': [2020, 2021, 2022, 2023],
    'experience_level': ['SE', 'MI', 'EN', 'EX'],
    'job_title': ['Data Science & Machine Learning', 'Data Engineering & Infrastructure', 'Data Analysis & Analytics', 'Data Management', 'Artificial Intelligence (AI)', 'Business Intelligence (BI)'],
    'employee_residence': ['North America', 'Europe', 'Other'],
    'remote_ratio': [0, 50, 100],
    'company_location': ['North America', 'Europe', 'Other'],
    'company_size': ['S', 'M', 'L']
}
  
    # Predict salary
    prediction = pipeline.predict(input_data_transformed)
    st.success(f'Predicted Salary: {prediction[0]}')
