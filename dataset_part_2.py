#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np


# In[3]:


df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
df.head(10)


# In[4]:


df.isnull().sum()/len(df)*100


# In[5]:


df.dtypes


# In[8]:


df['LaunchSite'].value_counts()


# In[9]:


df['Orbit'].value_counts()


# In[10]:


landing_outcomes = df['Outcome'].value_counts()


# In[11]:


landing_outcomes


# In[12]:


for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)


# In[13]:


bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes


# In[15]:


# Create a list to store success indicators (0 for bad, 1 for good)
landing_class = []

# Iterate through each outcome in the 'Outcome' column
for outcome in df['Outcome']:
  # Check if the outcome is in the bad_outcome set
  if outcome in bad_outcomes:
    landing_class.append(0)  # Append 0 for bad outcome
  else:
    landing_class.append(1)  # Append 1 for successful outcome

# Assign the list to the landing_class variable
df['landing_class'] = landing_class


# In[17]:


df['Class']=landing_class
df[['Class']].head(8)


# In[18]:


df.head(5)


# In[19]:


df.columns


# In[20]:


del df['landing_class']


# In[21]:


df.head(5)


# In[22]:


df.to_csv(r"C:\Users\arjun\Documents\DataScience\Assignment\dataset_part_2.csv", index=False)


# In[25]:


ccafs_df = df[df['LaunchSite'] == "CCAFS SLC 40"]


# In[26]:


class_counts = ccafs_df['Class'].value_counts()


# In[27]:


class_counts


# In[ ]:




