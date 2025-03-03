#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler


# In[10]:


# Load Dataset
file_path = "D:\Entri Assignments\Employee.csv"
df = pd.read_csv(file_path)


# # Data Exploration

# ### List Unique Values in Each Feature

# In[11]:


print("Unique values in each feature:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
    
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# In[12]:


# Renaming Columns
df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]


# # Data Cleaning

# In[13]:


print("\nMissing Values:")
print(df.isnull().sum())

df.replace({'0': np.nan, 0: np.nan}, inplace=True)  # Replace 0 with NaN in all columns

df.drop_duplicates(inplace=True)  # Removing duplicate rows



# In[14]:


# Handling Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical missing values with mode



# In[15]:


# Detecting and Handling Outliers
for col in df.select_dtypes(include=[np.number]):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), df[col].median(), df[col])



# # Data Analysis

# In[16]:


filtered_df = df[(df['age'] > 40) & (df['salary'] < 5000)]
print("Filtered Data (Age > 40 & Salary < 5000):")
print(filtered_df.head())



# In[17]:


# Plot Age vs Salary
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['age'], y=df['salary'])
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary')
plt.show()



# In[18]:


# Count of people per place
plt.figure(figsize=(8, 5))
df['place'].value_counts().plot(kind='bar')
plt.xlabel('Place')
plt.ylabel('Count')
plt.title('Number of People from Each Place')
plt.show()


# # Data Encoding

# In[19]:


label_encoders = {}
for col in df.select_dtypes(include=['object']):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le



# In[20]:


# Feature Scaling
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

scaled_standard = scaler_standard.fit_transform(df.select_dtypes(include=[np.number]))
scaled_minmax = scaler_minmax.fit_transform(df.select_dtypes(include=[np.number]))




# In[21]:


# Convert back to DataFrame
df_standard = pd.DataFrame(scaled_standard, columns=df.select_dtypes(include=[np.number]).columns)
df_minmax = pd.DataFrame(scaled_minmax, columns=df.select_dtypes(include=[np.number]).columns)

print("\nPreprocessing Complete! The dataset is ready for Machine Learning.")


# In[ ]:





# In[ ]:





# In[ ]:




