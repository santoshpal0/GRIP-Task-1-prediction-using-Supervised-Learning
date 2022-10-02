#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with Python Scikit Learn
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# # Simple Linear Regression
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
# 

# In[31]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[34]:


df=pd.read_csv("C:\\Users\\user\\Desktop\\students.csv")
df.head()


# In[9]:


df.shape


# In[8]:


df.describe()


# In[19]:


df.info()


# In[21]:


#To check null values
df.isnull().sum()


# In[10]:


# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:


# In[22]:


# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ##### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score

# ### Preparing the data
# #### The next step is to divide the data into "attributes" (inputs) and "labels" (outputs)

# In[12]:


X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[16]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# #### Training the Algorithm
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[17]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[24]:


#create object
lr = LinearRegression()
#fitting the model
lr.fit(X_train,y_train)


# In[26]:


print(lr.coef_, lr.intercept)


# In[18]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[27]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[28]:


# Comparing Actual vs Predicted
dtc = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
dtc 


# In[29]:


#Let's predict the score for 9.25 hpurs
print('Score of student who studied for 9.25 hours a dat', regressor.predict([[9.25]]))


# In[ ]:





# In[33]:


#Checking the efficiency of model
mean_squ_error = mean_squared_error(y_test, y_pred[:5])
mean_abs_error = mean_absolute_error(y_test, y_pred[:5])
print("Mean Squred Error:",mean_squ_error)
print("Mean absolute Error:",mean_abs_error)


# In[ ]:





# In[ ]:




