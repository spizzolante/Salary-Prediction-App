#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd


# In[126]:


data = pd.read_csv("modeling_data.csv")


# In[138]:


numerical_columns = ['salary_in_usd']


# In[140]:


categorical_columns = ['work_year', 'experience_level', 'job_title', 'employee_residence', 'remote_ratio', 'company_location','company_size']


# In[141]:


scaler = MinMaxScaler()


# In[130]:


scaler.fit(data[numerical_columns])


# In[131]:


regressor = RandomForestRegressor(n_estimators=100, random_state=42) 


# In[148]:


categories = [
    [2020, 2021, 2022, 2023],  # work_year
    ['SE', 'MI', 'EN', 'EX'],           # experience_level
    ['Data Science & Machine Learning', 'Data Engineering & Infrastructure', 'Data Analysis & Analytics', 'Data Management', 'Artificial Intelligence (AI)', 'Business Intelligence (BI)'],  # job_title
    ['North America', 'Europe', 'Other'],  # employee_residence
    [0, 50, 100],                       # remote_ratio
    ['North America', 'Europe', 'Other'],  # company_location
    ['S', 'M', 'L']                       # company_size
]


# In[135]:


#encoder = OneHotEncoder(categories=[categories[column] for column in categories])


# In[149]:


#ct = ColumnTransformer(
#    transformers=[
#        ('encoder', encoder, categorical_columns)
#    ],
#    remainder='drop'  # Drop columns not explicitly transformed
#)

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(categories=categories), categorical_columns),
        ('scaler', MinMaxScaler(), numerical_columns)
    ],
    remainder='passthrough'
)


# In[150]:


ct.fit(data)


# In[151]:


pipeline = Pipeline(steps=[('preprocessor', ct),
                           ('regressor', regressor)])


# In[146]:


#display(scaler)


# In[152]:


with open('model.pkl', 'wb') as file:
    pickle.dump((scaler, ct, pipeline), file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




