#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data = pd.read_csv("C:\\Users\\dell\\Downloads\\Housing.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[12]:


X = data[['area', 'bedrooms']]  # Features
y = data['price']  # Target variable


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[16]:


y_pred = model.predict(X_test)


# In[17]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')


# In[18]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()


# In[21]:


plt.plot(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()


# In[ ]:




