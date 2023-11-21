#!/usr/bin/env python
# coding: utf-8

# # PRODIGY INFOTECH 
# 
# ## TASK-2
# 
# ###  Perform data cleaning and exploratory data analysis(EDA) on a dataset of your choice, such as the Titanic dataset from kaggle. Explore the relationships between variables and identify patterns and trends in the data.

# In[44]:


#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[45]:


#Load the dataset of titanic
Data=pd.read_csv("C:/Users/steph/OneDrive/Documents/titanic.csv")
print(Data)


# In[46]:


# Create a dataframe for the data
df=pd.DataFrame(Data)
df


# In[47]:


#First 5 rows of the dataset
df.head()


# In[48]:


#Last 5 rows of the dataset
df.tail()


# In[49]:


#Check for all the columns of the dataset
df.columns


# In[50]:


#Check for the number of rows and columns of the dataset
df.shape


# In[51]:


#Check for the information ,i.e, dtype and null value for each column
df.info


# In[52]:


#Check for Statistical Analysis
df.describe


# In[53]:


#Check for the null values
print(df.isnull().sum())


# ## DATA CLEANING

# In[54]:


# Impute missing values with mean for numerical variables and mode for categorical variables
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

#Drop irrelevant columns
df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

#Check for the missing values
print(df.isnull().sum())


# ## EXPLORATORY DATA ANALYSIS

# ### 1. Survival rate by Passenger class

# In[65]:


sns.barplot(x='Pclass', y='Survived',data=df)
plt.title('Survival Rate by Passenger class')
plt.show()


# ### 2. Survival rate by Sex

# In[57]:


sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()


# ### 3. Survival rate by Age

# In[74]:


sns.histplot(x='Age', hue='Survived', data=df)
plt.title('Survival Rate by Age')
plt.show()


# ### 4. Survival rate by Number of Siblings or Spouses Abroad

# In[67]:


sns.barplot(x='SibSp', y='Survived', data=df)
plt.title('Survival Rate by Number of Siblings or Spouses Abroad')
plt.show()


#  ### 5. Survival rate by Number of Parents or Children Abroad

# In[75]:


sns.barplot(x='Parch', y='Survived', data=df)
plt.title('Survival Rate by Number of Parents or Children Abroad')
plt.show()


# ### 6. Survival rate by Embarked

# In[64]:


sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Embarked')
plt.show()

