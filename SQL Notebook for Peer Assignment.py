#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install sqlalchemy==1.3.9')


# In[5]:


get_ipython().system('pip install ipython-sql')


# In[16]:


get_ipython().run_line_magic('load_ext', 'sql')


# In[17]:


import csv, sqlite3

con = sqlite3.connect("my_data1.db")
cur = con.cursor()


# In[18]:


get_ipython().system('pip install -q pandas==1.1.5')


# In[19]:


get_ipython().run_line_magic('sql', 'sqlite:///my_data1.db')


# In[20]:


import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")


# In[21]:


get_ipython().run_line_magic('sql', 'create table SPACEXTABLE as select * from SPACEXTBL where Date is not null')


# In[22]:


get_ipython().run_line_magic('sql', 'select distinct "Launch_Site" from SPACEXTBL')


# Task 1
# 
# Display the names of the unique launch sites in the space mission

# In[23]:


get_ipython().run_line_magic('sql', 'select distinct "Launch_Site" from SPACEXTBL')


# Task 2
# 
# Display 5 records where launch sites begin with the string 'CCA'

# In[24]:


get_ipython().run_line_magic('sql', 'select * from SPACEXTBL where  "Launch_Site" like \'CCA%\' limit 5')


# In[55]:


get_ipython().run_line_magic('sql', 'select Customer, sum(PAYLOAD_MASS__KG_) as "Total_Payload_Mass" from SPACEXTBL group by "Customer"')


# Task 3
# 
# Display the total payload mass carried by boosters launched by NASA (CRS)

# In[61]:


get_ipython().run_cell_magic('sql', '', 'SELECT Customer, SUM(PAYLOAD_MASS__KG_) AS "Total_Payload_Mass" \nFROM SPACEXTBL \nWHERE Customer LIKE \'NASA (CRS)\' \nGROUP BY "Customer"\n')


# Task 4
# 
# Display average payload mass carried by booster version F9 v1.1

# In[70]:


get_ipython().run_cell_magic('sql', '', 'SELECT Booster_Version, AVG(PAYLOAD_MASS__KG_) AS "Average Payload Mass"\nFROM SPACEXTBL\nWHERE Booster_Version LIKE \'F9 v1.1\'\nGROUP BY \'Booster_Version\'\n')


# Task 5
# 
# List the date when the first succesful landing outcome in ground pad was acheived.

# In[75]:


get_ipython().run_cell_magic('sql', '', 'SELECT MIN(Date) AS "First_Ground_Pad_Success_Date"\nFROM SPACEXTBL\nWHERE Landing_Outcome LIKE \'Success (ground pad)\'\n')


# Task 6
# 
# List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000

# In[2]:


get_ipython().run_cell_magic('sql', '', "SELECT Booster_Version\nFROM SPACEXTBL\nWHERE Landing_Outcome LIKE 'Success (drone ship)'\n  AND PAYLOAD_MASS__KG_ > 4000\n  AND PAYLOAD_MASS__KG_ < 6000\n")


# Task 7
# 
# List the total number of successful and failure mission outcomes¶

# In[3]:


get_ipython().run_cell_magic('sql', '', 'SELECT Landing_Outcome, COUNT(*) AS Total_Missions\nFROM SPACEXTBL\nGROUP BY Landing_Outcome\n')


# Task 8
# 
# List the names of the booster_versions which have carried the maximum payload mass. Use a subquery

# In[79]:


get_ipython().run_cell_magic('sql', '', 'SELECT Booster_Version\nFROM SPACEXTBL\nWHERE PAYLOAD_MASS__KG_ = (\n  SELECT MAX(PAYLOAD_MASS__KG_)\n  FROM SPACEXTBL\n);\n')


# Task 9
# 
# List the records which will display the month names, failure landing_outcomes in drone ship ,booster versions, launch_site for the months in year 2015.¶

# In[81]:


get_ipython().run_cell_magic('sql', '', "SELECT  \n  CASE\n    -- Extract month number (1-12) and convert it to month name\n    WHEN SUBSTR(Date, 6, 2) = '01' THEN 'January'\n    WHEN SUBSTR(Date, 6, 2) = '02' THEN 'February'\n    WHEN SUBSTR(Date, 6, 2) = '03' THEN 'March'\n    WHEN SUBSTR(Date, 6, 2) = '04' THEN 'April'\n    WHEN SUBSTR(Date, 6, 2) = '05' THEN 'May'\n    WHEN SUBSTR(Date, 6, 2) = '06' THEN 'June'\n    WHEN SUBSTR(Date, 6, 2) = '07' THEN 'July'\n    WHEN SUBSTR(Date, 6, 2) = '08' THEN 'August'\n    WHEN SUBSTR(Date, 6, 2) = '09' THEN 'September'\n    WHEN SUBSTR(Date, 6, 2) = '10' THEN 'October'\n    WHEN SUBSTR(Date, 6, 2) = '11' THEN 'November'\n    WHEN SUBSTR(Date, 6, 2) = '12' THEN 'December'\n  END AS Month,\n  Landing_Outcome,\n  Booster_Version,\n  Launch_Site\nFROM SPACEXTBL\nWHERE Landing_Outcome LIKE 'Failure (drone ship)'\n  AND SUBSTR(Date, 0, 5) = '2015';\n\n")


# Task 10
# 
# Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.¶

# In[83]:


get_ipython().run_cell_magic('sql', '', "WITH RankedOutcomes AS (\n  SELECT Landing_Outcome, COUNT(*) AS Outcome_Count,\n         DENSE_RANK() OVER (ORDER BY COUNT(*) DESC) AS Rank\n  FROM SPACEXTBL\n  WHERE Date >= '2010-06-04' AND Date <= '2017-03-20'\n  GROUP BY Landing_Outcome\n)\nSELECT Landing_Outcome, Outcome_Count, Rank\nFROM RankedOutcomes\nORDER BY Rank;\n")


# In[71]:


get_ipython().run_line_magic('sql', 'select * from SPACEXTBL')


# In[ ]:




