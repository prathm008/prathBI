#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[2]:


tesla = pd.read_csv("C:/Users/prath/Downloads/tesla.csv")


# In[3]:


tesla.head()


# In[4]:


tesla.info()


# In[7]:


tesla['Date'] = pd.to_datetime(tesla['Date'])


# In[13]:


print(f'Dataframe contains stock prices between {tesla.Date.min()} {tesla.Date.max()}')
print(f'Total days = {(tesla.Date.max() - tesla.Date.min()).days} days')


# In[11]:


tesla.describe()


# In[15]:


tesla[['Open','High','Low','Close','Adj Close','Volume']].plot(kind= 'box')


# In[16]:


# Setting the layout for our plot
layout = go.Layout(
    title='Stock Prices of Tesla',
    xaxis=dict( 
       title='Date',
       titlefont=dict(
           family='Courier New, monospace',
           size=18,
        color='#7f7f7f'
)

), 
    yaxis=dict(
    title='Price',
    titlefont=dict(
    family='Courier New, monospace', 
        size=18, 
        color='#7f7f7f'
      )
    )
)

tesla_data = [{'x':tesla['Date'], 'y':tesla['Close']}] 
plot = go.Figure(data= tesla_data, layout=layout)


# In[17]:


#plot(plot) #plotting offline
iplot(plot)


# In[18]:


#Building the regression model
from sklearn.model_selection import train_test_split

#For preprocessing

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#For model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[19]:


X = np.array(tesla.index).reshape(-1, 1)
Y = tesla['Close']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)


# In[20]:


#Featue scaling
scaler = StandardScaler().fit(X_train)


# In[21]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[22]:


#Plot actual and predicted values for train dataset
trace0 = go.Scatter( 
    x = X_train.T[0],
    y = Y_train, 
    mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)
tesla_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=tesla_data,layout=layout)


# In[23]:


iplot(plot2)


# In[24]:


#Calculate scores for model evaluation
scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test,lm.predict(X_test))}
'''
print(scores)

