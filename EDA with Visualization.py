#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns


# In[2]:


import requests
from js import fetch
import io


# In[4]:


df = pd.read_csv(r"C:\Users\arjun\Documents\DataScience\Assignment\dataset_part_2.csv")


# In[5]:


df


# In[7]:


sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()


# ### TASK 1: Visualize the relationship between Flight Number and Launch Site

# In[8]:


plt.figure(figsize=(2,12))
sns.catplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df, aspect = 2)
plt.xlabel("Flight Number")
plt.ylabel("Launch Site")
plt.title("Relationship between Flight Number and Launch Site")
plt.tight_layout()
plt.show()


# # Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value

# In[9]:


plt.scatter(data=df, x="FlightNumber", y="LaunchSite", c=df["Class"], cmap="viridis", alpha=0.7)
plt.xlabel("Flight Number")
plt.ylabel("Launch Site")
plt.title("Relationship between Flight Number and Launch Site")
plt.show()


# ### TASK 2: Visualize the relationship between Payload and Launch Site

# In[10]:


sns.catplot(data=df, x="PayloadMass", y="LaunchSite", hue="Class", aspect = 2)
plt.xlabel("Payload Mass")
plt.ylabel("Launch Site")
plt.title("Relationship between Payload Mass and Launch Site")
plt.show()


# In[11]:


plt.scatter(data=df, x="PayloadMass", y="LaunchSite", c=df["Class"], cmap="viridis", alpha=0.7)
plt.xlabel("Payload Mass")
plt.ylabel("Launch Site")
plt.title("Relationship between Payload Mass and Launch Site")
plt.show()


# ### TASK  3: Visualize the relationship between success rate of each orbit type

# In[12]:


df_orbit = df.groupby('Orbit')["Class"].mean().reset_index()
sns.barplot(x=df_orbit["Orbit"], y=df_orbit["Class"])
plt.xlabel("Orbit")
plt.ylabel("Sucess Rate")
plt.title("Relationship between success rate of each orbit type")
plt.show()


# ### TASK  4: Visualize the relationship between FlightNumber and Orbit type

# In[13]:


plt.scatter(data=df, x="FlightNumber", y="Orbit", c=df["Class"], alpha=0.7)  # Adjust alpha for transparency
plt.xlabel("Flight Number")
plt.ylabel("Orbit")
plt.title("Scatter Plot of Flight Numbers vs. Orbits")
plt.show()


# ### TASK  5: Visualize the relationship between Payload and Orbit type

# In[14]:


plt.scatter(data=df, x="PayloadMass", y="Orbit", c=df["Class"], alpha=0.7)  # Adjust alpha for transparency
plt.xlabel("Payload Mass")
plt.ylabel("Orbit")

plt.show()


# ### TASK  6: Visualize the launch success yearly trend

# In[15]:


year=[]
def Extract_year():
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year
Extract_year()
df['Date'] = year
df.head()


# In[16]:


Yearly_Data = df.groupby('Date')['Class'].mean().reset_index()
Yearly_Data


# In[17]:


sns.lineplot(x=Yearly_Data["Date"], y=Yearly_Data["Class"])
plt.xlabel("Year")
plt.ylabel("Success Rate")

plt.show()


# In[18]:


features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()


# ### TASK  7: Create dummy variables to categorical columns

# In[19]:


# Define categorical columns for One-Hot Encoding
categorical_cols = ['Orbit', 'LaunchSite', 'LandingPad', 'Serial']

# Apply One-Hot Encoding using get_dummies
features_one_hot = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

# Display the first few rows of the encoded DataFrame
print(features_one_hot.head())


# In[20]:


features_one_hot


# ### TASK  8: Cast all numeric columns to `float64`

# In[ ]:


features_one_hot = features_one_hot.astype('float64')


# In[ ]:


features_one_hot


# In[ ]:


features_one_hot.to_csv(r"C:\Users\arjun\Documents\DataScience\Assignment\dataset_part_3.csv", index=False)


# In[ ]:




